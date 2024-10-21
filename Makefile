py-format:
	rye run ruff check python/rux_2024/ --select I --fix
	rye run ruff format python/rux_2024/
py-test:
	rye run pytest -vv --pyargs python/rux_2024/
py-lint:
	rye run ruff check python/rux_2024/
py-static:
	rye run mypy python/rux_2024/

rs-format:
	cargo fmt
rs-test:
	cargo test
rs-lint:
	cargo clippy -- -D warnings

build:
	maturin develop --skip-install

test: rs-test py-test
check: rs-lint py-lint py-static
prepare: build rs-format py-format test check

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
