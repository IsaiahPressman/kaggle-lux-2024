py-format:
	rye run ruff check python/ --select I --fix
	rye run ruff format python/
py-test:
	rye run pytest -vv --pyargs python/
py-lint:
	rye run ruff check python/
py-static:
	rye run mypy python/
py-prepare: py-format py-test py-lint py-static

rs-format:
	cargo +nightly fmt
rs-test:
	cargo test
rs-test-full:
	cargo test -- --include-ignored
rs-lint:
	cargo clippy -- -D warnings
rs-prepare: rs-format rs-test rs-lint

build:
	maturin develop --skip-install

build-release:
	maturin develop --release

test: rs-test-full py-test
check: rs-lint py-lint py-static
prepare: build rs-format py-format test check

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
