py-format:
	rye run ruff check python/ --select I --fix
	rye run ruff format python/
py-test:
	CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu rye run pytest -vv -m "not slow" --pyargs python/
py-test-full:
	CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu rye run pytest -vv --pyargs python/
py-lint:
	rye run ruff check python/
py-static:
	rye run mypy python/
py-prepare: py-format py-lint py-static py-test

rs-format:
	cargo +nightly fmt
rs-test:
	cargo test
rs-test-full:
	cargo test -- --include-ignored
rs-lint:
	cargo clippy --all-targets -- -D warnings
rs-prepare: rs-format rs-lint rs-test

build:
	maturin develop --skip-install

build-release:
	maturin develop --release

test: rs-test-full py-test-full
check: rs-lint py-lint py-static
prepare: build-release rs-format py-format check test

clean:
	cargo clean
	rm -rf .mypy_cache .pytest_cache .ruff_cache
