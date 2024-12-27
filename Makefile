py-format:
	rye run ruff check python/ --select I --fix
	rye run ruff format python/
py-test:
	CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu rye run pytest -vv -m "not agent and not slow" --pyargs python/
py-test-slow:
	CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu rye run pytest -vv -m "not agent" --pyargs python/
py-test-agent:
	CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu rye run pytest -vv -m "agent" --pyargs python/
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
	maturin develop

build-release:
	maturin develop --release

test: rs-test-full py-test-slow
check: rs-lint py-lint py-static
prepare: build rs-format py-format check test
# TODO: build-release for agent
prepare-agent: prepare py-test-agent

clean:
	cargo clean
	rm -rf .mypy_cache .pytest_cache .ruff_cache
