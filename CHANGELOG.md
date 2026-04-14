# Changelog

## [0.6.1](https://github.com/doublewordai/autobatcher/compare/autobatcher-v0.6.0...autobatcher-v0.6.1) (2026-04-14)


### Bug Fixes

* strip non-serializable kwargs ([9a6f94c](https://github.com/doublewordai/autobatcher/commit/9a6f94ca3b9c32acc8bd7c5fe46a518dd221e46c))
* strip non-serializable kwargs ([1dc0019](https://github.com/doublewordai/autobatcher/commit/1dc0019b808f1441ab5b5f3542096fb822d47402))

## [0.6.0](https://github.com/doublewordai/autobatcher/compare/autobatcher-v0.5.0...autobatcher-v0.6.0) (2026-04-14)


### Features

* **serve:** emit structured batch lifecycle events ([#21](https://github.com/doublewordai/autobatcher/issues/21)) ([90cfe52](https://github.com/doublewordai/autobatcher/commit/90cfe522108257c34ef7e6ff1709a4328d176075))

## [0.5.0](https://github.com/doublewordai/autobatcher/compare/autobatcher-v0.4.1...autobatcher-v0.5.0) (2026-04-14)


### Features

* subclass AsyncOpenAI for compat with MAF ([69a84bf](https://github.com/doublewordai/autobatcher/commit/69a84bf5603f60e2adf7705497edff9adf87c27d))
* subclass AsyncOpenAI for compat with MAF ([52232a0](https://github.com/doublewordai/autobatcher/commit/52232a0ea34d14c4084547596067661992f6130b))


### Bug Fixes

* lint ([829931c](https://github.com/doublewordai/autobatcher/commit/829931cdab61ef93bfe083f064b8ae2fdaf4da8c))
* tests ([a566c22](https://github.com/doublewordai/autobatcher/commit/a566c227f7dab6acc6acf5c62e78bf9a66f62e3b))

## [0.4.1](https://github.com/doublewordai/autobatcher/compare/autobatcher-v0.4.0...autobatcher-v0.4.1) (2026-04-13)


### Bug Fixes

* **cli:** allow arbitrary completion window strings ([#17](https://github.com/doublewordai/autobatcher/issues/17)) ([20dca1c](https://github.com/doublewordai/autobatcher/commit/20dca1c322bd8d54e1e93d2dd1cb21723463a431))

## [0.4.0](https://github.com/doublewordai/autobatcher/compare/autobatcher-v0.3.1...autobatcher-v0.4.0) (2026-04-13)


### Features

* add with_raw_response support ([7e76bad](https://github.com/doublewordai/autobatcher/commit/7e76bad680ddbccd9f9f2ed1101b407b0b27bf60))


### Documentation

* clarify with_raw_response example in _RawResponseWrapper docstring ([57b0cbf](https://github.com/doublewordai/autobatcher/commit/57b0cbfa5930b89f490389eaa4c7c36e10713f52))

## [0.3.1](https://github.com/doublewordai/autobatcher/compare/autobatcher-v0.3.0...autobatcher-v0.3.1) (2026-03-16)


### Bug Fixes

* handle streaming requests through batch API ([#12](https://github.com/doublewordai/autobatcher/issues/12)) ([a378759](https://github.com/doublewordai/autobatcher/commit/a37875913032245a0058054d7c57fde9160ed174))

## [0.3.0](https://github.com/doublewordai/autobatcher/compare/autobatcher-v0.2.0...autobatcher-v0.3.0) (2026-03-16)


### Features

* add serve mode for OpenAI-compatible HTTP proxy ([#10](https://github.com/doublewordai/autobatcher/issues/10)) ([356d4e5](https://github.com/doublewordai/autobatcher/commit/356d4e5ba5bd9460404cf2ddf10369b6c4751da2))

## [0.2.0](https://github.com/doublewordai/autobatcher/compare/autobatcher-v0.1.1...autobatcher-v0.2.0) (2026-03-03)


### Features

* add embeddings and responses API support, increase defaults ([#9](https://github.com/doublewordai/autobatcher/issues/9)) ([2846e66](https://github.com/doublewordai/autobatcher/commit/2846e66d5679ba14423a23f2ff025f8814028650))
* commits should produce minor version bumps, not patch bumps. ([f302dea](https://github.com/doublewordai/autobatcher/commit/f302dea622fb6fdc5828d99e6360ed1d3d85f0f2))


### Bug Fixes

* remove bump-patch-for-minor-pre-major from release-please config ([f302dea](https://github.com/doublewordai/autobatcher/commit/f302dea622fb6fdc5828d99e6360ed1d3d85f0f2))

## [0.1.1](https://github.com/doublewordai/autobatcher/compare/autobatcher-v0.1.0...autobatcher-v0.1.1) (2026-01-05)


### Features

* add release-please and PyPI trusted publishing workflows ([aea341d](https://github.com/doublewordai/autobatcher/commit/aea341d57d6fbee30f186547691bf6d1ff08f69d))


### Documentation

* update readme ([64d8d24](https://github.com/doublewordai/autobatcher/commit/64d8d24640e18aa456d50a9a7b25d091bb7332a5))
