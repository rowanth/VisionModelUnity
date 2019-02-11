Shader "Custom/Depth" {
	SubShader{
		Tags{ "RenderType" = "Opaque" }
		LOD 200

		Pass{

			CGPROGRAM

				#pragma vertex vert
				#pragma fragment frag

				#include "UnityCG.cginc"

				struct VertexInput {
					float4 v : POSITION;
				};
				
				float4 _g_AgentPositions[500];

				struct VertexOutput {
					float4 pos : SV_POSITION;
					float depth : DEPTH;
				};

				VertexOutput vert(VertexInput v) {
					VertexOutput o;
					o.pos = UnityObjectToClipPos(v.v);
					o.depth = -UnityObjectToViewPos(v.v).z * _ProjectionParams.w;

					return o;
				}

				float4 frag(VertexOutput o) : SV_TARGET {
					return fixed4(_g_AgentPositions[0].xyz,1);
				}
				/*
				float4 frag(VertexOutput o) : SV_TARGET {
					float invert = 1 - o.depth;
					return fixed4(o.depth, o.depth, o.depth, 1);
				}
				*/

			ENDCG
		}
	}
}