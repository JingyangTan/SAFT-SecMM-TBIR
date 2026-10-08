# Third-party components

SAFT uses the following third-party components. Each component is subject to its upstream license.

## Apple MobileCLIP

The SAFT adapter for [MobileCLIP](https://github.com/apple-aiml-research/ml-mobileclip) is located in `src/models/mobileclip/`. The MobileCLIP source is included in `third_party/ml-mobileclip-main/`. Install it with:

```bash
pip install ./third_party/ml-mobileclip-main
```



## TinyCLIP

The [TinyCLIP](https://github.com/wkcn/TinyCLIP) implementation is included in `src/models/TinyCLIP/`, along with its upstream license and documentation.

## OpenCLIP

[OpenCLIP](https://github.com/mlfoundations/open_clip) is installed through the `open_clip_torch` Python package.