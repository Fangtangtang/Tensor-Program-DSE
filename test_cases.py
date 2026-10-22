from typing import List

from general_dag_scheduler import scheduler, Einsum

single_kernel_einsum = [
    Einsum(
        operand_dims={
            "A": ("i", "j"),
            "W1": ("j", "k"),
            "B": ("i", "k"),
        },
        output_operand="B",
        dim_sizes={"i": 1024, "j": 512, "k": 256},
    )
]

# Test Case 1: Branching DAG (Shared Input)
# Computes two independent paths starting from a shared input A.
# A(i, j) * W1(j, k) -> B(i, k)
# A(i, j) * W2(j, l) -> C(i, l)
shared_input_einsums = [
    Einsum(
        operand_dims={
            "A": ("i", "j"),
            "W1": ("j", "k"),
            "B": ("i", "k"),
        },
        output_operand="B",
        dim_sizes={"i": 1024, "j": 512, "k": 256, "l": 128},
    ),
    Einsum(
        operand_dims={
            "A": ("i", "j"),
            "W2": ("j", "l"),
            "C": ("i", "l"),
        },
        output_operand="C",
        dim_sizes={"i": 1024, "j": 512, "k": 256, "l": 128},
    ),
]

# Test Case 2: Skip-Connection / ResNet Block
# An input X is projected to a different channel size, projected again, and then added back to X.
# X(h, w, c) * W1(c, d) -> Y(h, w, d)
# Y(h, w, d) * W2(d, c) -> Z(h, w, c)
# Z(h, w, c) + X(h, w, c) -> Out(h, w, c) (elementwise skip-addition)
skip_connection_einsums = [
    Einsum(
        operand_dims={
            "X": ("h", "w", "c"),
            "W1": ("c", "d"),
            "Y": ("h", "w", "d"),
        },
        output_operand="Y",
        dim_sizes={"h": 64, "w": 64, "c": 128, "d": 256},
    ),
    Einsum(
        operand_dims={
            "Y": ("h", "w", "d"),
            "W2": ("d", "c"),
            "Z": ("h", "w", "c"),
        },
        output_operand="Z",
        dim_sizes={"h": 64, "w": 64, "c": 128, "d": 256},
    ),
    Einsum(
        operand_dims={
            "Z": ("h", "w", "c"),
            "X": ("h", "w", "c"),
            "Out": ("h", "w", "c"),
        },
        output_operand="Out",
        dim_sizes={"h": 64, "w": 64, "c": 128, "d": 256},
    ),
]

# Test Case 3: Multi-Stage Reduction
# Compresses channel dimensions and then sequentially reduces spatial, then channel dimensions.
# X(n, c, h, w) * W(d, c) -> Y(n, d, h, w)
# Y(n, d, h, w) -> Z(n, d)  (spatial sum reduction over h, w)
# Z(n, d) -> Out(n)         (channel sum reduction over d)
multi_stage_reduction_einsums = [
    Einsum(
        operand_dims={
            "X": ("n", "c", "h", "w"),
            "W": ("d", "c"),
            "Y": ("n", "d", "h", "w"),
        },
        output_operand="Y",
        dim_sizes={"n": 32, "c": 64, "h": 16, "w": 16, "d": 128},
    ),
    Einsum(
        operand_dims={
            "Y": ("n", "d", "h", "w"),
            "Z": ("n", "d"),
        },
        output_operand="Z",
        dim_sizes={"n": 32, "c": 64, "h": 16, "w": 16, "d": 128},
    ),
    Einsum(
        operand_dims={
            "Z": ("n", "d"),
            "Out": ("n",),
        },
        output_operand="Out",
        dim_sizes={"n": 32, "c": 64, "h": 16, "w": 16, "d": 128},
    ),
]

# Test Case 4: Attention Projection Block (5 Einsums)
# Computes Q, K, V projections from X. Then contracts Q and K to get S (attention score).
# Finally contracts S and V to compute attention output O.
# X(b, s, h) * WQ(h, d) -> Q(b, s, d)
# X(b, s, h) * WK(h, d) -> K(b, s, d)
# X(b, s, h) * WV(h, d) -> V(b, s, d)
# Q(b, s, d) * K(b, s, d) -> S(b, s, s)  (contracts d)
# S(b, s, s) * V(b, s, d) -> O(b, s, d)  (contracts s)
attention_block_einsums = [
    Einsum(
        operand_dims={
            "X": ("b", "s", "h"),
            "WQ": ("h", "d"),
            "Q": ("b", "s", "d"),
        },
        output_operand="Q",
        dim_sizes={"b": 4, "s": 256, "h": 512, "d": 64},
    ),
    Einsum(
        operand_dims={
            "X": ("b", "s", "h"),
            "WK": ("h", "d"),
            "K": ("b", "s", "d"),
        },
        output_operand="K",
        dim_sizes={"b": 4, "s": 256, "h": 512, "d": 64},
    ),
    Einsum(
        operand_dims={
            "X": ("b", "s", "h"),
            "WV": ("h", "d"),
            "V": ("b", "s", "d"),
        },
        output_operand="V",
        dim_sizes={"b": 4, "s": 256, "h": 512, "d": 64},
    ),
    Einsum(
        operand_dims={
            "Q": ("b", "s", "d"),
            "K": ("b", "s", "d"),
            "S": ("b", "s", "s"),
        },
        output_operand="S",
        dim_sizes={"b": 4, "s": 256, "h": 512, "d": 64},
    ),
    Einsum(
        operand_dims={
            "S": ("b", "s", "s"),
            "V": ("b", "s", "d"),
            "O": ("b", "s", "d"),
        },
        output_operand="O",
        dim_sizes={"b": 4, "s": 256, "h": 512, "d": 64},
    ),
]


def attention_block_einsums_breakq(b=1, s=256, s1=256, h=512, d=64):
    return [
        Einsum(
            operand_dims={
                "X": ("b", "s", "h"),
                "WQ": ("h", "d"),
                "Q": ("b", "s", "d"),
            },
            output_operand="Q",
            dim_sizes={"b": b, "s": s, "h": h, "d": d},
            accel_gran={"s": 512, "h": 128, "d": 128},
            compute_cost=70710,
            operation="matmul",
        ),
        Einsum(
            operand_dims={
                "X": ("b", "s", "h"),
                "WK": ("h", "d"),
                "K": ("b", "s", "d"),
            },
            output_operand="K",
            dim_sizes={"b": b, "s": s, "h": h, "d": d},
            accel_gran={"s": 512, "h": 128, "d": 128},
            compute_cost=70710,
            operation="matmul",
        ),
        Einsum(
            operand_dims={
                "X": ("b", "s", "h"),
                "WV": ("h", "d"),
                "V": ("b", "s", "d"),
            },
            output_operand="V",
            dim_sizes={"b": b, "s": s, "h": h, "d": d},
            accel_gran={"s": 128, "h": 128, "d": 512},
            compute_cost=70710,
            operation="matmul",
        ),
        Einsum(
            operand_dims={
                "Q": ("b", "s1", "d"),
                "K": ("b", "s", "d"),
                "S": ("b", "s1", "s"),
            },
            output_operand="S",
            dim_sizes={"b": b, "s1": s1, "s": s, "d": d},
            accel_gran={"s": 128, "s1": 512, "d": 128},
            compute_cost=70710,
            operation="matmul",
        ),
        Einsum(
            operand_dims={
                "S": ("b", "s1", "s"),
                "V": ("b", "s", "d"),
                "O": ("b", "s1", "d"),
            },
            output_operand="O",
            dim_sizes={"b": b, "s1": s1, "s": s, "d": d},
            accel_gran={"s1": 512, "s": 128, "d": 128},
            compute_cost=70710,
            operation="matmul",
        ),
    ]


# Test Case 5: Transformer Feed-Forward Network
# Projects input X to a larger hidden dimension, applies an elementwise activation, then projects back.
# X(b, s, h) * W1(h, d_ff) -> Y(b, s, d_ff)
# Y(b, s, d_ff) -> Y_act(b, s, d_ff) (elementwise activation)
# Y_act(b, s, d_ff) * W2(d_ff, h) -> Z(b, s, h)
feed_forward_einsums = [
    Einsum(
        operand_dims={
            "X": ("b", "s", "h"),
            "W1": ("h", "d_ff"),
            "Y": ("b", "s", "d_ff"),
        },
        output_operand="Y",
        dim_sizes={"b": 4, "s": 256, "h": 512, "d_ff": 2048},
    ),
    Einsum(
        operand_dims={
            "Y": ("b", "s", "d_ff"),
            "Y_act": ("b", "s", "d_ff"),
        },
        output_operand="Y_act",
        dim_sizes={"b": 4, "s": 256, "h": 512, "d_ff": 2048},
    ),
    Einsum(
        operand_dims={
            "Y_act": ("b", "s", "d_ff"),
            "W2": ("d_ff", "h"),
            "Z": ("b", "s", "h"),
        },
        output_operand="Z",
        dim_sizes={"b": 4, "s": 256, "h": 512, "d_ff": 2048},
    ),
]


# Test Case 6: Llama-style Feed-Forward Network (SwiGLU)
# Uses a gate projection and an up projection, multiplies them, then projects down.
# X(bs, h) * W_up(h, d_ff) -> Y_up(bs, d_ff)
# X(bs, h) * W_gate(h, d_ff) -> Y_gate(bs, d_ff)
# Y_gate(bs, d_ff) -> Y_act(bs, d_ff) (elementwise activation)
# Y_up(bs, d_ff) * Y_act(bs, d_ff) -> Y_mid(bs, d_ff) (elementwise multiplication)
# Y_mid(bs, d_ff) * W_down(d_ff, h) -> Z(bs, h)
def llama_ffn_einsums(bs=128, h=2880, d_ff=2880, num_exp=1):
    return [
        Einsum(
            operand_dims={
                "X": ("bs", "h"),
                "W_up": ("h", "d_ff", "e"),
                "Y_up": ("bs", "d_ff", "e"),
            },
            output_operand="Y_up",
            dim_sizes={"bs": bs, "h": h, "d_ff": d_ff, "e": num_exp},
            accel_gran={"bs": 128, "h": 128, "d_ff": 512, "e": 1},
            compute_cost=70710,
            operation="matmul",
        ),
        Einsum(
            operand_dims={
                "X": ("bs", "h"),
                "W_gate": ("h", "d_ff", "e"),
                "Y_gate": ("bs", "d_ff", "e"),
            },
            output_operand="Y_gate",
            dim_sizes={"bs": bs, "h": h, "d_ff": d_ff, "e": num_exp},
            accel_gran={"bs": 128, "h": 128, "d_ff": 512, "e": 1},
            compute_cost=70710,
            operation="matmul",
        ),
        # Einsum(
        #     operand_dims={
        #         'Y_gate': ('bs', 'd_ff'),
        #         'Y_act': ('bs', 'd_ff'),
        #     },
        #     output_operand='Y_act',
        #     dim_sizes={'bs': 128, 'd_ff': 2880},
        #     accel_gran={'PRODUCT': 128},
        #     compute_cost=100,
        #     operation='sigmoid(alpha*clamp(Y_gate))'
        # ),
        Einsum(
            operand_dims={
                "Y_up": ("bs", "d_ff", "e"),
                "Y_gate": ("bs", "d_ff", "e"),
                "Y_mid": ("bs", "d_ff", "e"),
            },
            output_operand="Y_mid",
            dim_sizes={"bs": bs, "d_ff": d_ff, "e": num_exp},
            accel_gran={"PRODUCT": 128},
            compute_cost=44,
            operation="elemwise_mul(linear_gate(Y_up),sigmoid(alpha*clamp(Y_gate)))",
        ),
        Einsum(
            operand_dims={
                "Y_mid": ("bs", "d_ff", "e"),
                "W_down": ("d_ff", "h1", "e"),
                "Z": ("bs", "h1"),
            },
            output_operand="Z",
            dim_sizes={"bs": bs, "h1": h, "d_ff": d_ff, "e": num_exp},
            accel_gran={"bs": 128, "h1": 512, "d_ff": 128, "e": 1},
            compute_cost=70710,
            operation="matmul",
        ),
    ]


# Test Case 7: Attention decode
def attention_decode_einsums(b=1, s=256, s1=1, h=128, d=512):
    return [
        Einsum(
            operand_dims={
                "X": ("b", "s1", "h"),
                "WQ": ("h", "d"),
                "Q": ("b", "s1", "d"),
            },
            output_operand="Q",
            dim_sizes={"b": b, "s1": s1, "h": h, "d": d},
            accel_gran={"b": 512, "s1": 1, "h": 128, "d": 128},
            compute_cost=100,
            operation="matmul",
        ),
        Einsum(
            operand_dims={
                "Q": ("b", "s1", "d"),
                "K": ("b", "s", "d"),
                "S": ("b", "s1", "s"),
            },
            output_operand="S",
            dim_sizes={"b": b, "s1": s1, "s": s, "d": d},
            accel_gran={"b": 512, "s1": 1, "s": 128, "d": 128},
            compute_cost=100,
            operation="matmul",
        ),
        Einsum(
            operand_dims={
                "S": ("b", "s1", "s"),
                "V": ("b", "s", "d"),
                "O": ("b", "s1", "d"),
            },
            output_operand="O",
            dim_sizes={"b": b, "s1": s1, "s": s, "d": d},
            accel_gran={"b": 512, "s1": 1, "s": 128, "d": 128},
            compute_cost=100,
            operation="matmul",
        ),
    ]


# Test Case 8: KDA core decode
def kda_core_decode_einsums(d_k=128, d_v=128, heads=96):
    # S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
    # =
    # K3 numbers are as given: d_k=128, d_v=128, heads=96
    return [
        Einsum(
            operand_dims={"k": ("d_k0", "h"), "b": ("h",), "bk": ("d_k0", "h")},
            output_operand="bk",
            dim_sizes={"h": heads, "d_k0": d_k},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="elew_mul",
        ),
        Einsum(
            operand_dims={
                "bk": ("d_k0", "h"),
                "k": ("d_k1", "h"),
                "a": ("d_k1", "h"),
                "1bkka": ("d_k0", "d_k1", "h"),
            },
            output_operand="1bkka",
            dim_sizes={"h": heads, "d_k0": d_k, "d_k1": d_k},
            accel_gran={"PRODUCT": 128},
            compute_cost=4 * 22,
            operation="(I - b k k^T) Diag(a)",
        ),
        Einsum(
            operand_dims={
                "1bkka": ("d_k0", "d_k1", "h"),
                "S_{t-1}": ("d_k1", "d_v", "h"),
                "1bkkas": ("d_k0", "d_v", "h"),
            },
            output_operand="1bkkas",
            dim_sizes={"h": heads, "d_k0": d_k, "d_k1": d_k, "d_v": d_v},
            accel_gran={"d_k0": 128, "d_k1": 128, "d_v": 128, "h": 1},
            compute_cost=70710,
            operation="matmul",
        ),
        Einsum(
            operand_dims={
                "bk": ("d_k0", "h"),
                "v_T": ("d_v", "h"),
                "bkv": ("d_k0", "d_v", "h"),
            },
            output_operand="bkv",
            dim_sizes={"h": heads, "d_k0": d_k, "d_v": d_v},
            accel_gran={"PRODUCT": 128},
            compute_cost=2 * 22,
            operation="elew_mul",
        ),
        Einsum(
            operand_dims={
                "1bkkas": ("d_k0", "d_v", "h"),
                "bkv": ("d_k0", "d_v", "h"),
                "S_t": ("d_k0", "d_v", "h"),
            },
            output_operand="S_t",
            dim_sizes={"h": heads, "d_k0": d_k, "d_v": d_v},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="elew_add",
        ),
    ]


def lcan_einsums(name, hidden=7168, d_out=128, d_out_name="d_k", heads=96, norm=True):
    ret = [
        Einsum(
            operand_dims={
                "hidden": ("hidden",),
                name + "_lin_w": ("hidden", d_out_name, "h"),
                name + "_lin_out": (d_out_name, "h"),
            },
            output_operand=name + "_lin_out",
            dim_sizes={"hidden": hidden, d_out_name: d_out, "h": heads},
            accel_gran={"hidden": 128, d_out_name: 128, "h": 1},
            compute_cost=70710,
            operation="matmul",
        ),
        Einsum(
            operand_dims={
                name + "_history": (d_out_name, "h", "hist"),
                name + "_conv_w": (d_out_name, "hist"),
                name + "_partial_conv": (d_out_name, "h"),
            },
            output_operand=name + "_partial_conv",
            dim_sizes={"hist": 3, d_out_name: d_out, "h": heads},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="matmul",
        ),
        Einsum(
            operand_dims={
                name + "_lin_out": (d_out_name, "h"),
                name + "_conv_w_0": (d_out_name,),
                name + "_lin_out_scaled": (d_out_name, "h"),
            },
            output_operand=name + "_lin_out_scaled",
            dim_sizes={d_out_name: d_out, "h": heads},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="elew_mul",
        ),
        Einsum(
            operand_dims={
                name + "_lin_out_scaled": (d_out_name, "h"),
                name + "_partial_conv": (d_out_name, "h"),
                name + "_conv_out": (d_out_name, "h"),
            },
            output_operand=name + "_conv_out",
            dim_sizes={d_out_name: d_out, "h": heads},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="elew_add",
        ),
        Einsum(
            operand_dims={
                name + "_conv_out": (d_out_name, "h"),
                name + "_act_out" if norm else name: (d_out_name, "h"),
            },
            output_operand=name + "_act_out" if norm else name,
            dim_sizes={d_out_name: d_out, "h": heads},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="silu",
        ),
    ]
    if norm:
        ret.append(
            Einsum(
                operand_dims={
                    name + "_act_out": (d_out_name, "h"),
                    name + "_computed_norm": ("h",),
                },
                output_operand=name + "_computed_norm",
                dim_sizes={d_out_name: d_out, "h": heads},
                accel_gran={"PRODUCT": 128},
                compute_cost=22,
                operation="l2_norm_compute",
            )
        )
        ret.append(
            Einsum(
                operand_dims={
                    name + "_act_out": (d_out_name, "h"),
                    name + "_computed_norm": ("h",),
                    name: (d_out_name, "h"),
                },
                output_operand=name,
                dim_sizes={d_out_name: d_out, "h": heads},
                accel_gran={"PRODUCT": 128},
                compute_cost=22,
                operation="l2_norm_divide",
            )
        )
    return ret


# Test Case 9: KDA Full Decode
def kda_full_decode_einsums(hidden=7168, d_k=128, d_v=128, heads=96, a_rank=128):
    # (mostly from diagram, not actual code)
    # qk: linear->shortconv->act->norm
    # v: linear->shortconv->act
    # a: linear_dn->linear_up->sigmoid
    # b: linear_dn->sigmoid
    # bypass: linear_dn->linear_up->sigmoid
    # output=linear(norm(KDA_core(qk, v, a, b)) * bypass)
    einsums = []
    einsums += lcan_einsums(
        "q", hidden=hidden, d_out=d_k, d_out_name="d_k1", heads=heads, norm=True
    )
    einsums += lcan_einsums(
        "k", hidden=hidden, d_out=d_k, d_out_name="d_k0", heads=heads, norm=True
    )
    einsums += lcan_einsums(
        "v", hidden=hidden, d_out=d_v, d_out_name="d_v", heads=heads, norm=False
    )
    einsums += [  # a and b
        Einsum(
            operand_dims={
                "hidden": ("hidden",),
                "W_a_down": ("hidden", "a_rank", "h"),
                "a_down": ("a_rank", "h"),
            },
            output_operand="a_down",
            dim_sizes={"hidden": hidden, "a_rank": a_rank, "h": heads},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="matvec",
        ),
        Einsum(
            operand_dims={
                "a_down": ("a_rank",),
                "W_a_up": ("a_rank", "d_k1", "h"),
                "a": ("d_k1", "h"),
            },
            output_operand="a",
            dim_sizes={"a_rank": a_rank, "h": heads, "d_k1": d_k},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="sigmoid(matvec)",
        ),
        Einsum(
            operand_dims={
                "hidden": ("hidden",),
                "W_b_down": ("hidden", "h"),
                "b": ("h",),
            },
            output_operand="b",
            dim_sizes={"hidden": hidden, "h": heads},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="sigmoid(matvec)",
        ),
    ]
    einsums += kda_core_decode_einsums(d_k=d_k, d_v=d_v, heads=heads)
    einsums += [  # S_{t-1}q -> RMSNorm
        Einsum(
            operand_dims={
                "S_{t-1}": ("d_k1", "d_v", "h"),
                "q": ("d_k1", "h"),
                "o": ("d_v", "h"),
            },
            output_operand="o",
            dim_sizes={"d_k1": d_k, "d_v": d_v, "h": heads},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="matvec",
        ),
        Einsum(
            operand_dims={"o": ("d_v", "h"), "computed_o_rmsnorm": ("h",)},
            output_operand="computed_o_rmsnorm",
            dim_sizes={"d_v": d_v, "h": heads},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="rmsnorm",
        ),
        Einsum(
            operand_dims={
                "o": ("d_v", "h"),
                "computed_o_rmsnorm": ("h",),
                "o_normed": ("d_v", "h"),
            },
            output_operand="o_normed",
            dim_sizes={"d_v": d_v, "h": heads},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="rmsnorm_divide",
        ),
    ]
    einsums += [  # bypass
        Einsum(
            operand_dims={
                "hidden": ("hidden",),
                "W_g_down": ("hidden", "a_rank", "h"),
                "g_down": ("a_rank", "h"),
            },
            output_operand="g_down",
            dim_sizes={"hidden": hidden, "a_rank": a_rank, "h": heads},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="matvec",
        ),
        Einsum(
            operand_dims={
                "g_down": ("a_rank",),
                "W_g_up": ("a_rank", "d_v", "h"),
                "g": ("d_v", "h"),
            },
            output_operand="g",
            dim_sizes={"a_rank": a_rank, "h": heads, "d_v": d_v},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="sigmoid(matvec)",
        ),
        Einsum(
            operand_dims={
                "o_normed": ("d_v", "h"),
                "g": ("d_v", "h"),
                "o_bypassed": ("d_v", "h"),
            },
            output_operand="o_bypassed",
            dim_sizes={"d_v": d_v, "h": heads},
            accel_gran={"PRODUCT": 128},
            compute_cost=22,
            operation="elew_mul",
        ),
    ]
    return einsums


def einsum_by_einsum_scheduler(
    einsums: List[Einsum],
    capacity: int,
    num_cores: int = 1,
    enforce_optimal_placement: bool = True,
    allow_spilling: bool = False,
):
    total_compute = 0
    total_comm = 0
    for e in einsums:
        sm, solver = scheduler(
            [e],
            capacity,
            num_cores,
            enforce_optimal_placement,
            allow_spilling,
            debug=False,
        )
        total_compute += solver.Value(sm.final_compute_cost)
        total_comm += solver.Value(sm.final_communication_cost)
    print("FINAL COMPUTE AND COMMUNICATION", total_compute, total_comm)


if __name__ == "__main__":
    print("=========================================")
    print("Running Test Case 0: Single Node")
    print("=========================================")
    scheduler(single_kernel_einsum, capacity=512 * 1024, allow_spilling=True)

    print("=========================================")
    print("Running Test Case 1: Shared Input (Branching DAG)")
    print("=========================================")
    # Using small capacity to force tile decisions
    scheduler(shared_input_einsums, capacity=512 * 1024, allow_spilling=True)

    print("=========================================")
    print("Running Test Case 2: Skip-Connection / ResNet Block")
    print("=========================================")
    scheduler(skip_connection_einsums, capacity=1024 * 1024, allow_spilling=True)

    print("=========================================")
    print("Running Test Case 3: Multi-Stage Reduction")
    print("=========================================")
    scheduler(multi_stage_reduction_einsums, capacity=256 * 1024, allow_spilling=True)

    print("=========================================")
    print("Running Test Case 4: Attention Projection Block (break q)")
    print("=========================================")
    scheduler(
        attention_block_einsums_breakq(b=32, s=1 * 1024, s1=1 * 1024, h=4096, d=512),
        num_cores=1,
        capacity=3 * 1024 * 1024,
        allow_spilling=True,
    )
    einsum_by_einsum_scheduler(
        attention_block_einsums_breakq(b=32, s=1 * 1024, s1=1 * 1024, h=4096, d=512),
        num_cores=1,
        capacity=3 * 1024 * 1024,
        allow_spilling=True,
    )

    print("=========================================")
    print("Running Test Case 5: Transformer Feed-Forward Network")
    print("=========================================")
    scheduler(feed_forward_einsums, capacity=512 * 1024, allow_spilling=True)

    print("=========================================")
    print("Running Test Case 6: Llama-style Feed-Forward Network (SwiGLU)")
    print("=========================================")
    scheduler(
        llama_ffn_einsums(1024, 7168, 3072, num_exp=6),
        capacity=6 * 1024 * 1024,
        num_cores=1,
        allow_spilling=True,
    )
    einsum_by_einsum_scheduler(
        llama_ffn_einsums(1024, 7168, 3072, num_exp=6),
        capacity=6 * 1024 * 1024,
        num_cores=1,
        allow_spilling=True,
    )
    scheduler(
        llama_ffn_einsums(64 * 1024, 1024, 2048, num_exp=1),
        capacity=3 * 1024 * 1024,
        num_cores=8,
        allow_spilling=True,
    )
    einsum_by_einsum_scheduler(
        llama_ffn_einsums(16 * 1024, 1024, 2048, num_exp=1),
        capacity=3 * 1024 * 1024,
        num_cores=8,
        allow_spilling=True,
    )

    print("=========================================")
    print("Running Test Case 7: Attention Decode")
    print("=========================================")
    scheduler(
        attention_decode_einsums(b=512, s=128 + 1024, s1=1, h=128, d=512),
        capacity=3 * 1024 * 1024,
        allow_spilling=True,
    )

    print("=========================================")
    print("Running Test Case 8: KDA Core Decode")
    print("=========================================")
    scheduler(
        kda_core_decode_einsums(d_k=128, d_v=128, heads=96),
        capacity=3 * 1024 * 1024,
        allow_spilling=True,
    )

    print("=========================================")
    print("Running Test Case 9: KDA Full Decode")
    print("=========================================")
    scheduler(
        kda_full_decode_einsums(hidden=7168, d_k=128, d_v=128, heads=96, a_rank=128),
        capacity=3 * 1024 * 1024,
        allow_spilling=True,
    )
