# rux-2024

### Setup instructions
1. [Install rust](https://www.rust-lang.org/tools/install) and add nightly toolchain for rustfmt:
   1. `rustup update nightly`
   2. `rustup component add rustfmt --toolchain nightly`
2. [Install rye](https://rye.astral.sh/guide/installation/)
2. Install maturin: `rye install maturin`
3. Install packages: `rye sync`
4. Activate venv: `. .venv/bin/activate`
5. If everything is working, `make prepare` should run  without errors
