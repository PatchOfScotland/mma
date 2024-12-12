import matplotlib.pyplot as plt
import numpy as np

def matmul():    
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

def large_mmm():
    ds = np.array([1024, 2048, 4096, 2048, 4096, 8192])
    ks = np.array([1024, 2048, 4096, 1024, 2048, 2048])
    time_tc_us = np.array([171, 951, 6413, 481, 3186, 12231])
    time_us_f16 = np.array([284, 1830, 14600, 943, 7170, 29544])
    time_us_f32 = np.array([351, 2476, 21515, 1247, 10857, 44688])
    total_ops = ds * ds * ks * 2
    tflops_tc = total_ops / (time_tc_us * 1_000_000)    
    tflops_orig_f16 = total_ops / (time_us_f16 * 1_000_000)
    tflops_orig_f32 = total_ops / (time_us_f32 * 1_000_000)

    labels = [f"d={d}, k={k}" for (d,k) in zip (ds, ks)]

    plt.plot(tflops_tc, marker='o', linestyle='-', color='skyblue', label="CUDA backend + TC f16/f32")
    plt.plot(tflops_orig_f16, marker='o', linestyle='-', color='red', label="CUDA backend f16")
    plt.plot(tflops_orig_f32, marker='o', linestyle='-', color='coral', label="CUDA backend f32")
    plt.title("Large matrix multiplication")
    plt.xlabel("Matrix size of $d\\times d \\times k$ matrix multiplication")
    plt.ylabel("TFLOPS")
    plt.xticks(range(len(time_tc_us)), labels=labels, rotation=45)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def lud_like():
    blocks = 256*256
    ds = np.array([16, 32, 64, 128])
    time_tc_us = np.array([154, 451, 1797, 10113])
    time_us_f32 = np.array([3088, 4791, 7461, 47724])
    time_us_f16 = np.array([3057, 4471, 6873, 39440])
    # time_us_f32 = np.array([1046, 6680, 68845, 1407986])
    total_ops = blocks * (ds * ds * ds * 2 + ds * ds)
    
    tflops_tc = total_ops / (time_tc_us * 1_000_000)
    tflops_orig_f16 = total_ops / (time_us_f16 * 1_000_000)
    tflops_orig_f32 = total_ops / (time_us_f32 * 1_000_000)

    plt.plot(tflops_tc, marker='o', linestyle='-', color='skyblue', label="CUDA backend + TC f16/f32")
    plt.plot(tflops_orig_f16, marker='o', linestyle='-', color='red', label="CUDA backend f16")
    plt.plot(tflops_orig_f32, marker='o', linestyle='-', color='coral', label="CUDA backend f32")
    plt.title("LUD Matmul")
    plt.xlabel("Matrix size $n$ of $n\\times n \\times n$ matrix multiplication")
    plt.ylabel("TFLOPS")
    plt.xticks(range(len(time_tc_us)), labels=ds)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def attention_like():
    blocks = 1024 * 256
    ds = np.array([16, 32, 64, 128])
    ks = np.array([16, 32, 64, 64])
    time_tc_us = np.array([214, 557, 3981, 11064])
    time_us_f32 = np.array([14100, 21990, 35660, 129815])
    time_us_f16 = np.array([12276, 17419, 25304, 151786])
    
    tflops_tc = blocks * (ds * ds * ks) * 2 / (time_tc_us * 1_000_000)    
    tflops_orig_f16 = blocks * (ds * ds * ks) * 2 / (time_us_f16 * 1_000_000)
    tflops_orig_f32 = blocks * (ds ** 3) * 2 * 2 / (time_us_f32 * 1_000_000)

    # plt.axhline(124, color="r", label="FlashAttention")
    plt.plot(tflops_tc, marker='o', linestyle='-', color='skyblue', label="CUDA backend + TC f16/f32")
    plt.plot(tflops_orig_f16, marker='o', linestyle='-', color='red', label="CUDA backend f16")
    plt.plot(tflops_orig_f32, marker='o', linestyle='-', color='coral', label="CUDA backend f32")
    plt.title("Flash Attention Like")
    plt.xlabel("Matrix size $n$ of $n\\times n \\times n$ matrix multiplication")
    plt.ylabel("TFLOPS")
    plt.xticks(range(len(time_tc_us)), labels=ds)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def custom_attention():
    blocks = 100_000
    ds = np.array([16, 32, 64, 128])
    time_tc_us = np.array([382, 414, 2545, 8410])
    time_us_f16 = np.array([833, 5088, 47410, 505696])
    time_us_f32 = np.array([1046, 6680, 68845, 1407986])
    
    tflops_tc = blocks * (ds ** 3) * 2 * 2 / (time_tc_us * 1_000_000)
    tflops_orig_f16 = blocks * (ds ** 3) * 2 * 2 / (time_us_f16 * 1_000_000)
    tflops_orig_f32 = blocks * (ds ** 3) * 2 * 2 / (time_us_f32 * 1_000_000)

    # plt.axhline(124, color="r", label="FlashAttention")
    plt.plot(tflops_tc, marker='o', linestyle='-', color='skyblue', label="CUDA backend + TC f16/f32")
    plt.plot(tflops_orig_f16, marker='o', linestyle='-', color='b', label="CUDA backend f16")
    plt.plot(tflops_orig_f32, marker='o', linestyle='-', color='lightcoral', label="CUDA backend f32")
    plt.axhline(124, color="r", label="Flash Attention 1")
    plt.title("Custom Flash Attention Like")
    plt.xlabel("Matrix size $n$ of $n\\times n \\times n$ matrix multiplication")
    plt.ylabel("TFLOPS")
    plt.xticks(range(len(time_tc_us)), labels=ds)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
lud_like()
large_mmm()
attention_like()    
matmul()
custom_attention()
