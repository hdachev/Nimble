[![License: MIT](https://img.shields.io/packagist/l/doctrine/orm.svg)](https://opensource.org/licenses/MIT)

# Nimble
A simple and lightweight OpenGL rendering engine built on top of dwSampleFramework.

## Features
* Forward and Deferred rendering paths.
* Physically Based Shading
* Image Based Lighting
* Cascaded Shadow Maps with Percentage Closer Filtering (PCF)
* Screen Space Ambient Occlusion (SSAO)
* Temporal Anti-Aliasing (TAA)
* Per-Object Motion Blur
* Screen Space Reflections
* Bloom
* Variety of Tone-Mapping operators (Reinhard, Uncharted 2, Filmic etc)
* GPU Profiling

## Screenshots

![Nimble](data/main1.jpg)

![Nimble](data/main2.jpg)

![Nimble](data/ssr.jpg)

![Nimble](data/mb1.jpg)

![Nimble](data/mb2.jpg)

![Nimble](data/ssao.jpg)

![Nimble](data/deferred.jpg)

## Dependencies
* [dwSampleFramework](https://github.com/diharaw/dwSampleFramework) 
* [json](https://github.com/nlohmann/json) 
* [nativefiledialog](https://github.com/mlabbe/nativefiledialog)

## License
```
Copyright (c) 2018 Dihara Wijetunga

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```