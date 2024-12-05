import matplotlib.pyplot as plt
import numpy as np

def matmul():
    attention_like = [((128 ** 3) * 2 * 2 * 10_000, 948)]
    matmul_names = ["Baseline",
                    "Swizzling",
                    "Vectorized copies,\nglobal to shared",
                    "Double buffering",
                    "Vectorized copies,\nshared registers",
                    "Asynchronous copies",
                    "3-stage pipelining",
                    "Cublas"]

    matmul = [44.524, 82.2175, 122.244, 131.552, 193.127, 205.562, 229.008, 255.012]
    plt.axhline(312, color = "r")
    plt.legend(["Peak performance"])
    bars = plt.bar(matmul_names, matmul, color='skyblue', edgecolor='black')
    # for bar in bars:
    #     bar.set_hatch("//")
    bars[-1].set_color("lightcoral")
    bars[-1].set_edgecolor("black")
    # plt.xlabel('Index', fontsize=14)
    plt.ylabel('TFLOPS', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def custom_attention():
    blocks = 100_000
    ds = np.array([16, 32, 64, 128])
    time_tc_us = np.array([382, 414, 2545, 9111])
    time_us_f16 = np.array([831, 5088, 47410, 505696])
    time_us_f32 = np.array([1046, 6680, 68845, 1407986])
    
    tflops_tc = blocks * (ds ** 3) * 2 * 2 / (time_tc_us * 1_000_000)
    tflops_orig_f16 = blocks * (ds ** 3) * 2 * 2 / (time_us_f16 * 1_000_000)
    tflops_orig_f32 = blocks * (ds ** 3) * 2 * 2 / (time_us_f32 * 1_000_000)
    
    plt.plot(tflops_tc, marker='o', linestyle='-', color='skyblue', label="CUDA backend + TC f16")
    plt.plot(tflops_orig_f16, marker='o', linestyle='-', color='r', label="CUDA backend f16")
    plt.plot(tflops_orig_f32, marker='o', linestyle='-', color='lightcoral', label="CUDA backend f32")
    plt.xlabel("Matrix size $n$ of $n\\times n \\times n$ matrix multiplication")
    plt.ylabel("TFLOPS")
    plt.xticks(range(len(time_tc_us)), labels=ds)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
matmul()
custom_attention()
