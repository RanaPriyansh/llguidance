//! Compare cancellable mask work after 4,096 to 4,351 byte visits.
//! Run with `cargo run -p llguidance --release --example cancellation -- cancel 30`.
//! Replace `cancel` with `baseline` for the uncancelled control.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Instant;

use anyhow::{bail, ensure, Result};
use llguidance::api::TopLevelGrammar;
use llguidance::earley::{BiasComputer, ParserRecognizer};
use llguidance::toktrie::{
    InferenceCapabilities, Recognizer, SimpleVob, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv,
};
use llguidance::{Matcher, ParserFactory};

const VOCAB_SIZE: usize = 128_000;
const CHECKPOINT: usize = 4_096;

struct Vocabulary(TokTrie);

impl TokenizerEnv for Vocabulary {
    fn tok_trie(&self) -> &TokTrie {
        &self.0
    }

    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.0.greedy_tokenize(bytes)
    }

    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}

fn vocabulary() -> TokEnv {
    let mut tokens: Vec<Vec<u8>> = (0..=255).map(|byte| vec![byte]).collect();
    for mut value in 0..VOCAB_SIZE - 257 {
        let mut token = vec![b'a'; 4];
        for byte in &mut token {
            *byte += (value % 26) as u8;
            value /= 26;
        }
        tokens.push(token);
    }
    tokens.push(b"<EOS>".to_vec());
    Arc::new(Vocabulary(TokTrie::from(
        &TokRxInfo::new(VOCAB_SIZE as u32, (VOCAB_SIZE - 1) as u32),
        &tokens,
    )))
}

// The public BiasComputer interface lets this example synchronize after real
// parser work without adding callbacks to the normal library execution path.
struct MeasuredBias {
    env: TokEnv,
    checkpoint: Arc<Barrier>,
    checkpoint_at: usize,
    visits: AtomicUsize,
}

struct MeasuredRecognizer<'a, 'b> {
    inner: &'a mut ParserRecognizer<'b>,
    checkpoint: &'a Barrier,
    checkpoint_at: usize,
    visits: usize,
}

impl Recognizer for MeasuredRecognizer<'_, '_> {
    fn cancellation_enabled(&self) -> bool {
        self.inner.cancellation_enabled()
    }

    fn cancellation_requested(&self) -> bool {
        self.inner.cancellation_requested()
    }

    fn pop_bytes(&mut self, count: usize) {
        self.inner.pop_bytes(count);
    }

    fn collapse(&mut self) {
        self.inner.collapse();
    }

    fn trie_started(&mut self, label: &str) {
        self.inner.trie_started(label);
    }

    fn trie_finished(&mut self) {
        self.inner.trie_finished();
    }

    fn try_push_byte(&mut self, byte: u8) -> bool {
        let allowed = self.inner.try_push_byte(byte);
        self.visits += 1;
        if self.visits == self.checkpoint_at {
            self.checkpoint.wait();
            self.checkpoint.wait();
        }
        allowed
    }

    fn save_stats(&mut self, visits: usize) {
        self.inner.save_stats(visits);
    }
}

impl BiasComputer for MeasuredBias {
    fn compute_bias(&self, inner: &mut ParserRecognizer<'_>, start: &[u8]) -> SimpleVob {
        let mut rec = MeasuredRecognizer {
            inner,
            checkpoint: &self.checkpoint,
            checkpoint_at: self.checkpoint_at,
            visits: 0,
        };
        let mut mask = self.trie().alloc_token_set();
        self.trie().add_bias(&mut rec, &mut mask, start);
        self.visits.store(rec.visits, Ordering::Relaxed);
        mask
    }

    fn trie(&self) -> &TokTrie {
        self.env.tok_trie()
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let cancel = match args.next().as_deref() {
        Some("cancel") => true,
        Some("baseline") => false,
        _ => bail!("usage: cancellation <cancel|baseline> [trials]"),
    };
    let trials = args
        .next()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(30);
    ensure!(
        trials > 0 && args.next().is_none(),
        "invalid trial count or extra argument"
    );

    let env = vocabulary();
    let mut factory = ParserFactory::new(&env, InferenceCapabilities::default(), &[])?;
    factory.quiet();
    let mut expected = env.tok_trie().alloc_token_set();
    for token in b'a' as TokenId..=b'z' as TokenId {
        expected.allow_token(token);
    }
    for token in 256..VOCAB_SIZE as TokenId {
        expected.allow_token(token);
    }
    println!("trial,cancelled,checkpoint_visits,request_us,remaining_us,byte_visits,drop_us");

    for trial in 0..trials {
        let checkpoint = Arc::new(Barrier::new(2));
        let checkpoint_at = CHECKPOINT + trial % 256;
        let bias = Arc::new(MeasuredBias {
            env: env.clone(),
            checkpoint: checkpoint.clone(),
            checkpoint_at,
            visits: AtomicUsize::new(0),
        });
        let mut parser =
            factory.create_parser(TopLevelGrammar::from_lark("start: /[a-z]+/".to_string()))?;
        parser.bias_computer = bias.clone();
        let mut matcher = Matcher::new(Ok(parser)).into_cancellable();
        let handle = matcher.cancellation_handle().unwrap();
        let worker = std::thread::spawn(move || {
            matcher.consume_token(b'a' as TokenId).unwrap();
            let result = matcher.compute_mask();
            (Instant::now(), result, matcher)
        });

        checkpoint.wait();
        let requested = Instant::now();
        if cancel {
            handle.cancel();
        }
        let request_us = requested.elapsed().as_secs_f64() * 1e6;
        checkpoint.wait();
        let (returned, result, matcher) = worker.join().expect("worker panicked");
        if cancel {
            ensure!(
                result.is_err() && matcher.is_cancelled(),
                "cancellation was not reported"
            );
        } else {
            let mask = result?;
            ensure!(mask == expected, "incorrect baseline mask");
        }
        let dropping = Instant::now();
        drop(matcher);
        let drop_us = dropping.elapsed().as_secs_f64() * 1e6;
        println!(
            "{trial},{cancel},{checkpoint_at},{request_us:.3},{:.3},{},{drop_us:.3}",
            returned.duration_since(requested).as_secs_f64() * 1e6,
            bias.visits.load(Ordering::Relaxed),
        );
    }
    Ok(())
}
