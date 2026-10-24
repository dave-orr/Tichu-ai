//! `tichu_engine::replay::build_all` for the browser. The page writes a game
//! log's JSON into memory from `alloc`, calls `build`, and reads the result
//! (replay data, or `{"error": ...}`) back as UTF-8 from `out_ptr`/`out_len`.

use std::cell::RefCell;

thread_local! {
    static OUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[no_mangle]
pub unsafe extern "C" fn dealloc(p: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(p, 0, len));
}

/// Builds the replays in the JSON at `p..p+len` (which it frees).
#[no_mangle]
pub unsafe extern "C" fn build(p: *mut u8, len: usize) {
    let input = Vec::from_raw_parts(p, len, len);
    let result = match std::str::from_utf8(&input).map_err(|e| e.to_string()).and_then(tichu_engine::replay::build_from_text) {
        Ok(r) if r.is_empty() => serde_json::json!({ "error": "no game logs in this file" }),
        Ok(r) => serde_json::Value::from(r),
        Err(e) => serde_json::json!({ "error": e }),
    };
    OUT.with(|o| *o.borrow_mut() = result.to_string().into_bytes());
}

#[no_mangle]
pub extern "C" fn out_ptr() -> *const u8 {
    OUT.with(|o| o.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn out_len() -> usize {
    OUT.with(|o| o.borrow().len())
}

/// The replay builder never draws entropy (decks come from the log); rand
/// just needs a source to link on wasm32.
fn no_entropy(_: &mut [u8]) -> Result<(), getrandom::Error> {
    Err(getrandom::Error::UNSUPPORTED)
}
getrandom::register_custom_getrandom!(no_entropy);
