# A Synthetic-Vision Based Local Collision Avoidance Model for Virtual Pedestrians (Unity Edition).

![teaser](Assets/Textures/teaser.png?raw=true)

## Background 
+ A synthetic-vision based steering approach for crowd simulation. https://dl.acm.org/citation.cfm?id=1778860
+ DAVIS: density-adaptive synthetic-vision based steering for virtual crowds. https://dl.acm.org/citation.cfm?id=2822030

## Requirements
+ Unity 2018.3.3f1 (64-bit) or higher
+ CUDA 9.2 Compatible Graphics Card
+ Windows (Although would be easy enough to port)

## Issues
+ This is something I have ported in the odd hour I had to spare, in order to evaluate Managed CUDA. I am sure there are lots of small errors dotted about.
+ One thing that slows this down in comparison to a pure OpenGL version is the fact that I am writing to a Unity RenderTarget and then having to read that back into a native Unity Texture2D (then convert to a CUDA float4 array!). This is not optimal. In pure OpenGL you write straight into a pixel buffer and then register it with CUDA. This is something I will have a look at when I have some time but its still pretty fast for small crowds.
+ Not very many Quality of Life features or UI functionality. These are features I will slowly add.
+ I am currently working on the Density filter, from the DAVIS paper mentioned above, and will push that out soon.
