use llguidance::{
    earley::{
        lexerspec::LexemeIdx,
        regexvec::{RegexVec, StateID},
    },
    toktrie::TokenId,
    Matcher,
};

#[test]
fn fast_forward_and_regexvec_methods_keep_public_signatures() {
    let _: fn(&mut Matcher) -> Vec<TokenId> = Matcher::compute_ff_tokens;
    let _: fn(&mut Matcher) -> Vec<TokenId> = Matcher::consume_ff_tokens;
    let _: fn(&mut Matcher) -> Vec<u8> = Matcher::compute_ff_bytes;
    let _: fn(&mut RegexVec, StateID, u8) -> StateID = RegexVec::transition;
    let _: fn(&mut RegexVec, StateID, LexemeIdx, u64) -> Result<bool, anyhow::Error> =
        RegexVec::check_subsume;
}
