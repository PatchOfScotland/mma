import matplotlib.pyplot as plt
import numpy as np



def matmul():
    matmul_names = ["Baseline",
                    "Swizzling",
                    "Vectorized copies,\nglobal to shared",
                    "Double buffering",
                    "ldmatrix",
                    "Asynchronous copies,\n3-stage pipelining",
                    "Cublas"]
    matmul = [41.4235, 70.2043, 128.67, 163.375, 233.442, 252.367, 253.18]
    plt.axhline(312, color = "r")
    plt.legend(["Hardware limit"])
    bars = plt.bar(matmul_names, matmul, color='skyblue', edgecolor='black')
    # for bar in bars:
    #     bar.set_hatch("//")
    bars[-1].set_color("lightcoral")
    bars[-1].set_edgecolor("black")
    # plt.xlabel('Index', fontsize=14)
    plt.ylabel('TFLOPS', fontsize=14)
    plt.xticks(rotation=60)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def lud_like():
    blocks = 256*256
    ds = np.array([16, 32, 64, 128])
    time_tc_us = np.array([154, 451, 1797, 10113])
    time_us_f32 = np.array([1737, 2617, 5629, 35165])
    time_us_f16 = np.array([1490, 2264, 4561, 29344])
    # time_us_f32 = np.array([1046, 6680, 68845, 1407986])
    total_ops = blocks * (ds * ds * ds * 2 + ds * ds)

    tflops_tc = total_ops / (time_tc_us * 1_000_000)
    tflops_orig_f16 = total_ops / (time_us_f16 * 1_000_000)
    tflops_orig_f32 = total_ops / (time_us_f32 * 1_000_000)

    plt.plot(ds, tflops_tc, marker='o', linestyle='-', color='coral', label="CUDA backend + TC f16/f32 mixed")
    plt.plot(ds, tflops_orig_f16, marker='o', linestyle='-', color='skyblue', label="CUDA backend f16")
    plt.plot(ds, tflops_orig_f32, marker='o', linestyle='-', color='b', label="CUDA backend f32")
    plt.title("LUD like, matrix multiplications of size $n\\times n \\times n$")
    plt.xlabel("$n$")
    plt.ylabel("TFLOPS")
    # plt.xticks(range(len(time_tc_us)), labels=ds)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def attention_like():
    blocks = 1024 * 256
    ds = np.array([16, 32, 64, 128])
    # Note size 128 does not fit in the k dimension
    ks = np.array([16, 32, 64, 64])
    time_tc_us_copy = np.array([214, 557, 3981, 11064])
    time_tc_us_no_copy = np.array([220, 528, 2555, 11608])
    time_tc_prot = np.array([167, 345, 1391, 5913])
    time_us_f32 = np.array([14100, 21990, 35660, 129815])
    time_us_f16 = np.array([12276, 17419, 25304, 151786])
    # NOTE: Below is with ks <= 64 and the above is ks < 128
    # time_us_f16

    tflops_tc_copy = blocks * (ds * ds * ks) * 2 / (time_tc_us_copy * 1_000_000)
    tflops_tc_no_copy = blocks * (ds * ds * ds) * 2 / (time_tc_us_no_copy * 1_000_000)
    tflops_tc_prot = blocks * (ds * ds * ds) * 2 / (time_tc_prot * 1_000_000)
    # NOTE these use k=128
    tflops_orig_f16 = blocks * (ds * ds * ds) * 2 / (time_us_f16 * 1_000_000)
    tflops_orig_f32 = blocks * (ds * ds * ds) * 2 / (time_us_f32 * 1_000_000)

    # plt.axhline(124, color="r", label="FlashAttention")
    plt.plot(ds, tflops_tc_copy, marker='o', linestyle='-', color='coral', label="CUDA backend + TC f16/f32 Futhark copy A")
    plt.plot(ds, tflops_tc_no_copy, marker='o', linestyle='-', color='red', label="CUDA backend + TC f16/f32 CuTe copy A")
    plt.plot(ds, tflops_tc_prot, marker='o', linestyle='-', color='g', label="Handwritten implementation using CuTe (without pipelining)")
    plt.plot(ds, tflops_orig_f16, marker='o', linestyle='-', color='skyblue', label="CUDA backend f16 Futhark copy A")
    plt.plot(ds, tflops_orig_f32, marker='o', linestyle='-', color='b', label="CUDA backend f32 Futhark copy A")
    plt.title("Flash Attention Like, matrix multiplications of size $n\\times n \\times k$")
    plt.xlabel("$n$")
    plt.ylabel("TFLOPS")
    #plt.xticks(range(len(time_tc_us)), labels=ds)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def custom_attention():
    blocks = 100_000
    ds = np.array([16, 32, 64, 128])
    time_tc_us = np.array([382, 414, 2545, 8410])
    time_tc_no_mbm_us = np.array([341, 398, 3894, 10858])
    time_us_f16 = np.array([837, 5189, 48471, 509904])
    time_us_f32 = np.array([1027, 6716, 69502, 1408956])

    tflops_tc = blocks * (ds ** 3) * 2 * 2 / (time_tc_us * 1_000_000)
    tflops_tc_no_mbm = blocks * (ds ** 3) * 2 * 2 / (time_tc_no_mbm_us * 1_000_000)
    tflops_orig_f16 = blocks * (ds ** 3) * 2 * 2 / (time_us_f16 * 1_000_000)
    tflops_orig_f32 = blocks * (ds ** 3) * 2 * 2 / (time_us_f32 * 1_000_000)

    # plt.axhline(124, color="r", label="FlashAttention")
    plt.plot(ds, tflops_tc, marker='o', linestyle='-', color='coral', label="CUDA backend + TC f16/f32 mixed")
    plt.plot(ds, tflops_tc_no_mbm, marker='o', linestyle='-', color='orange', label="CUDA backend + TC f16/f32 mixed no MBM")
    plt.plot(ds, tflops_orig_f16, marker='o', linestyle='-', color='skyblue', label="CUDA backend f16")
    plt.plot(ds, tflops_orig_f32, marker='o', linestyle='-', color='b', label="CUDA backend f32")
    # plt.axhline(124, color="magenta", label="Flash Attention 1")
    plt.title("Custom Flash Attention Like, matrix multiplications of size $n\\times n \\times n$")
    plt.xlabel("$n$")
    plt.ylabel("TFLOPS")
    # plt.xticks(range(len(time_tc_us)), labels=ds)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def batched_mmm():
    blocks = 32768
    ds = np.array([16, 32, 64, 128])
    time_tc_us = np.array([75, 215, 819, 3107])
    time_tc_prot = np.array([53, 199, 776, 3056])
    time_us_f16 = np.array([633, 1012, 2147, 14441])
    time_us_f32 = np.array([788, 1260, 2911, 17760])

    tflops_tc = blocks * (ds ** 3) * 2 / (time_tc_us * 1_000_000)
    tflops_tc_prot = blocks * (ds ** 3) * 2 / (time_tc_prot * 1_000_000)
    tflops_orig_f16 = blocks * (ds ** 3) * 2 / (time_us_f16 * 1_000_000)
    tflops_orig_f32 = blocks * (ds ** 3) * 2 / (time_us_f32 * 1_000_000)

    # plt.axhline(124, color="r", label="FlashAttention")
    plt.plot(ds, tflops_tc, marker='o', linestyle='-', color='coral', label="CUDA backend + TC f16/f32 mixed")
    plt.plot(ds, tflops_tc_prot, marker='o', linestyle='-', color='g', label="Handwritten implementation using CuTe (without pipelining)")
    plt.plot(ds, tflops_orig_f16, marker='o', linestyle='-', color='skyblue', label="CUDA backend f16")
    plt.plot(ds, tflops_orig_f32, marker='o', linestyle='-', color='b', label="CUDA backend f32")
    plt.title("Batched Matrix Multiplication, matrix multiplications of size $n\\times n \\times n$")
    plt.xlabel("$n$")
    plt.ylabel("TFLOPS")
    # plt.xticks(range(len(time_tc_us)), labels=ds)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def large_mmm():
    ds = np.array([1024, 2048, 4096, 8192])
    time_tc_us = np.array([169, 921, 6036, 45527])
    time_us_f16 = np.array([247, 1399, 10698, 86309])
    time_us_f32 = np.array([233, 1355, 10512, 86190])
    total_ops = ds ** 3 * 2
    tflops_tc = total_ops / (time_tc_us * 1_000_000)
    tflops_orig_f16 = total_ops / (time_us_f16 * 1_000_000)
    tflops_orig_f32 = total_ops / (time_us_f32 * 1_000_000)

    tflops_cute = [108.73,
                   187.55,
                   252.37,
                   238.49,
                   ]

    plt.plot(ds, tflops_tc, marker='o', linestyle='-', color='coral', label="CUDA backend + TC f16/f32 mixed")
    plt.plot(ds, tflops_orig_f16, marker='o', linestyle='-', color='skyblue', label="CUDA backend f16")
    # plt.plot(ds, tflops_cute, marker='o', linestyle='-', color='g', label="Handwritten implementation using CuTe")
    plt.plot(ds, tflops_orig_f32, marker='o', linestyle='-', color='b', label="CUDA backend f32")
    plt.title("Large Matrix Multiplication of size $n\\times n \\times n$")
    plt.xlabel("$n$")
    plt.ylabel("TFLOPS")
    # plt.xticks(range(len(time_tc_us)), labels=labels, rotation=45)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def prot_mmm1():
    ds = np.array([1024, 2048, 4096, 8192])
    tflops_pure = [54.96,
                   133.07,
                   170.22,
                   160.32,
                   ]
    tflops_cute = [108.73,
                   187.55,
                   252.37,
                   238.49,
                   ]
    tflops_cublas = [79.98,
                     198.84,
                     253.18,
                     253.50,
                     ]

    plt.axhline(312, color="r", label="Hardware limit")
    plt.plot(ds, tflops_pure, marker='o', linestyle='-', color='lime', label="Pure CUDA")
    plt.plot(ds, tflops_cute, marker='o', linestyle='-', color='green', label="Using CuTe building blocks")
    plt.plot(ds, tflops_cublas, marker='o', linestyle='-', color='darkviolet', label="Cublas")
    plt.title("Matrix Multiplication of size $n\\times n \\times n$")
    plt.xlabel("$n$")
    plt.ylabel("TFLOPS")
    # plt.xticks(range(len(time_tc_us)), labels=labels, rotation=45)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def prot_mmm2():
    ds = np.array([1024, 2048, 4096, 8192])
    tflops_pure = [135.33,
                   157.45,
                   170.22,
                   174.03,
                   ]
    tflops_cute = [204.16,
                   231.18,
                   252.37,
                   263.41,
                   ]
    tflops_cublas = [217.12,
                     234.42,
                     253.18,
                     259.99,
                     ]
    plt.axhline(312, color="r", label="Hardware limit")
    plt.plot(ds, tflops_pure, marker='o', linestyle='-', color='lime', label="Pure CUDA")
    plt.plot(ds, tflops_cute, marker='o', linestyle='-', color='green', label="Using CuTe building blocks")
    plt.plot(ds, tflops_cublas, marker='o', linestyle='-', color='darkviolet', label="Cublas")
    plt.title("Matrix Multiplication of size $4096\\times 4096 \\times k$")
    plt.xlabel("$k$")
    plt.ylabel("TFLOPS")
    # plt.xticks(range(len(time_tc_us)), labels=labels, rotation=45)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# matmul()
lud_like()
attention_like()
custom_attention()
batched_mmm()
large_mmm()
# prot_mmm1()
# prot_mmm2()
