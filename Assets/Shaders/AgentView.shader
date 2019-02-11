Shader "Unlit/AgentView"
{
	Properties
	{
		_MainTex("Texture", 2D) = "red" {}
	}
	SubShader
	{
		Tags{ "RenderType" = "Opaque" }
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag		
			#include "UnityCG.cginc"
			#pragma multi_compile_instancing
			#pragma enable_d3d11_debug_symbols

			struct Agent {
				float2x2 rot_mat;
				float2 vel_agent;
				float2 vel_goal;
				float pa;
				float ttg;
				uint id;
				uint a_id;
			};

			struct appdata {
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f {
				float2 uv : TEXCOORD0;
				float4 pos : SV_POSITION;

				float depth				: FLOAT0;
				float alpha				: FLOAT1;
				float alpha_dot			: FLOAT2;
				float speed				: FLOAT3;
				float speed_alpha		: FLOAT4;
				float speed_alpha_goal	: FLOAT5;

				Agent agent				: AGENT;
			};

			float4 _g_agent_rot_mat;
			float2 _g_agent_velocity;
			float2 _g_agent_velocity_goal;
			float _g_agent_pa;
			float _g_agent_ttg;
			int _g_agent_id;
			int _g_agent_aid;

			sampler2D _MainTex;
			float4 _MainTex_ST;
			StructuredBuffer<float2> positionBuffer;
			StructuredBuffer<float2> velocityBuffer;

			v2f vert(appdata v, uint instanceID : SV_InstanceID)
			{
				v2f o;
				
				//Create Agent (No UBO in Unity)
				Agent agent;
				agent.rot_mat = float2x2(_g_agent_rot_mat[0],_g_agent_rot_mat[1],_g_agent_rot_mat[2],_g_agent_rot_mat[3]);
				agent.vel_agent = _g_agent_velocity;
				agent.vel_goal = _g_agent_velocity_goal;
				agent.pa = _g_agent_pa;
				agent.ttg = _g_agent_ttg;
				agent.id = _g_agent_id;
				agent.a_id = _g_agent_aid;

				o.agent = agent;

				float4 newVertex = float4(0, 0, 0, 1);
				
				if (agent.a_id != instanceID)
				{
					float4 mm;
					const float alpha_a = 1.57f;
					o.speed_alpha = -10.0f;
					o.speed_alpha_goal = 0.0f;
					o.depth = 100.0f;

					float2 lpos_vertex = positionBuffer[agent.id + instanceID];

					if (agent.a_id >= 0) {
						mm = float4(agent.pa, 1.7f, agent.pa, 1.0f);
					}
					else {
						// Obstacles not scaled (i.e. walls, etc.)
						mm = float4(1.0f, 1.0f, 1.0f, 1.0f);
					}

					newVertex = mm * v.vertex + float4(lpos_vertex.x, 0.0f, lpos_vertex.y, 0.0f);

					float2 lvel_vertex = mul(agent.rot_mat, velocityBuffer[agent.id + instanceID]);
					float2 vel_composed = (lvel_vertex - agent.vel_agent);
					float2 vel_composed_goal = (lvel_vertex - agent.vel_goal);

					float2 rel_pos_vertex = UnityObjectToViewPos(newVertex).xz;
					rel_pos_vertex.y = -rel_pos_vertex.y;

					// Depth (to compute ttc)
					o.depth = length(rel_pos_vertex);
					float2 dir_c = normalize(rel_pos_vertex);
					float2 dir_cc = -dir_c;

					o.speed = length(lvel_vertex);

					// SpeedAlpha (to compute ttc)
					o.speed_alpha = dot(vel_composed, (dir_cc));
					o.speed_alpha_goal = dot(vel_composed_goal, (dir_cc));
					float2 dir_cn = normalize(rel_pos_vertex + vel_composed);

					float alpha_c = atan2(dir_c.y, dir_c.x);     // alpha(t)
					float alpha_cn = atan2(dir_cn.y, dir_cn.x);  // alpha(t+1s)

					// alpha
					o.alpha = alpha_c - alpha_a;
					if (o.alpha < -3.14) {
						o.alpha = o.alpha + 6.28;
					}

					o.alpha_dot = alpha_cn - alpha_c;
					if (o.alpha_dot < -3.14) {
						o.alpha_dot = o.alpha_dot + 6.28;
					}
					else if (o.alpha_dot > 3.14) {
						o.alpha_dot = o.alpha_dot - 6.28;
					}

					if (o.depth <= o.speed_alpha && abs(o.alpha_dot) > 1.57) {
						if (o.alpha_dot > 0.0) {
							o.alpha_dot = 3.14 - o.alpha_dot;
						}
						else {
							o.alpha_dot = -3.14 - o.alpha_dot;
						}
					}
				}

				o.pos = UnityObjectToClipPos(newVertex);
				o.uv = TRANSFORM_TEX(v.uv, _MainTex);

				return o;
			}

			float4 frag(v2f i) : COLOR 
			{
				const float a1 = 0.0f;
				const float b1 = 0.6f;
				const float c1 = 1.5f;

				float threshold = 1.57f;
				float theta_dot_1 = 0.0f;
				float theta_dot_2 = 0.0f;
				float ttc;
				float ttcg;
				float v;
				float Td = 0.0f;
				float first = 1.0f;
					
				float dist = max(0.0, i.depth - i.agent.pa);
				ttc = (i.speed_alpha != 0) ? dist / i.speed_alpha : 100;

				if (ttc >= 0.0 && ttc < 10.0)
				{
					v = a1 + b1 / pow(ttc, c1);
					Td = clamp(v, 0.0, threshold);
					ttcg = dist / i.speed_alpha_goal;

					if (ttc > 1.0f && ttcg >= 10.0f) {
						ttc = 10.0f;
					}

					if (abs(i.alpha_dot) < Td && (i.agent.ttg >= ttc || ttc < 1.0)) {
						theta_dot_1 = clamp((-Td + i.alpha_dot), -threshold, threshold);
						theta_dot_2 = clamp(( Td + i.alpha_dot), -threshold, threshold);

						first = i.alpha * i.alpha_dot;
					}
					else if (i.speed == 0.0 && (ttcg >= 0.0) && (ttcg < 1.0)) {
						// Not colliding: Check if directon to the goal is collision free.
						theta_dot_1 = 100.0;
						theta_dot_2 = -100.0;
						ttc = 5.0;
					}
				}
				return float4(theta_dot_1, theta_dot_2, ttc, first);
			}

			ENDCG
		}
	}
}