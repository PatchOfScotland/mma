import json
import matplotlib.pyplot as plt
import subprocess
import os

def mean(l):
    return sum(l) / len(l)

def nth(haystack, needle, n):
    parts= haystack.split(needle, n+1)
    return len(haystack)-len(parts[-1])-len(needle)

def plot_graph(title, xlabel, ylabel, plots):
    for x, y, t in plots:
        plt.plot(x, y, marker='o', linestyle='-', label=t)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    filename = title[:title.index(',')].replace(' ', '_') if ',' in title else title.replace(' ', '_')
    plt.savefig(os.path.join("graphs", f"{filename}.pdf"))

def run_command(backend, results_file, entry_point, script):
    args = [
        "futmma", 
        "bench", 
        f"--backend={backend}",
        "--pass-option=--nvrtc-option=-I../../cutlass/include" if backend=="cudatc" else "",
        f"--json={results_file}", 
        f"--entry-point={entry_point}", 
        script
    ]
    args = [a for a in args if a != ""]
    subprocess.run(args)

def assemble(experiment_list, blocks, result_func):
    to_plot = []
    results_file = "tmp.json"
    n_lookups = {
        16: 0,
        32: 1,
        64: 2,
        128: 3
    }
    for script, entry_point, backend, title in experiment_list:
        if type(entry_point) == list:
            tflops = [None]*len(entry_point)
            for ep in entry_point:
                run_command(backend, results_file, ep, script)

                with open(results_file, 'r') as json_file:
                    data = json.load(json_file)

                for ep, datasets in data.items():
                    for experiment, results in datasets["datasets"].items():
                        res, n = result_func(results, experiment, blocks)
                        tflops[n_lookups[n]] = res
            ns = [i for i in n_lookups.keys() if n_lookups[i] < len(tflops)]
            to_plot.append([ns, tflops, title])

        else:
            run_command(backend, results_file, entry_point, script)

            with open(results_file, 'r') as json_file:
                data = json.load(json_file)

            for entry_point, datasets in data.items():
                tflops = [None, None, None, None]
                for experiment, results in datasets["datasets"].items():
                    res, n = result_func(results, experiment, blocks)
                    tflops[n_lookups[n]] = res
                to_plot.append([list(n_lookups.keys()), tflops, title])
    os.remove(results_file)
    return to_plot
    
def batched_mmm_result(results, experiment, blocks):
    runtimes = results["runtimes"]
    mean_runtime = mean(runtimes)
    n = int(experiment[len(f"[{blocks}]["):nth(experiment, '[', 2)-1])
    tflops = (blocks * (n ** 3) * 2) / (mean_runtime * 1e6)
    return (tflops, n)

def batched_mmm():
    blocks = 32768

    experiment_list = [
#        ("batched_mmm_orig.fut", "mmmf16", "cuda", "CUDA backend f16"),
#        ("batched_mmm_orig.fut", "mmmf32", "cuda", "CUDA backend f32"),
        ("batched_mmm.fut", ["mmm_intra16", "mmm_intra32", "mmm_intra64", "mmm_intra128"], "cudatc", "CUDA backend w/ TC & mixed f16/f32")
    ]
    to_plot = assemble(experiment_list, blocks, batched_mmm_result)

    plot_graph("Batched Matrix Multiplication, matrix multiplications of size $n\\times n \\times n$", "$n$", "TFLOPS", to_plot)

def custom_attention_result(results, experiment, blocks):
    runtimes = results["runtimes"]
    mean_runtime = mean(runtimes)
    n = int(experiment[nth(experiment, ' ', 2)+1:nth(experiment, 'i32', 1)])
    tflops = (blocks * (n ** 3) * 2 * 2 ) / (mean_runtime * 1e6)
    return (tflops, n)

def custom_attention():
    blocks = 32768
    experiment_list = [
        ("custom_attention_like_orig.fut", "run_origf16", "cuda", "CUDA backend f16"),
        ("custom_attention_like_orig.fut", "run_origf32", "cuda", "CUDA backend f32"),
        ("custom_attention_like.fut", ["run16", "run32", "run64", "run128"], "cudatc", "CUDA backend w/ TC & mixed f16/f32")
    ]
    to_plot = assemble(experiment_list, blocks, custom_attention_result)

    plot_graph("Custom Flash Attention Like, matrix multiplications of size $n\\times n \\times n$", "$n$", "TFLOPS", to_plot)

def attention_result(results, experiment, blocks):
    runtimes = results["runtimes"]
    mean_runtime = mean(runtimes)
    if experiment.startswith("[1024]["):
        n = int(experiment[len(f"[1024]["):nth(experiment, '[', 2)-1])
    else:
        n = int(experiment[nth(experiment, ' ', 2)+1:nth(experiment, "i32", 2)])
    tflops = (blocks * (n * n * n) * 2 ) / (mean_runtime * 1e6)
    return (tflops, n)

def attention():
    blocks = 1024 * 256
    experiment_list = [
        ("attention_like_orig.fut", "run_no_intra_f16", "cuda", "CUDA backend f16 w/ Futhark copy A"),
        ("attention_like_orig.fut", "run_no_intra_f32", "cuda", "CUDA backend f32 w/ Futhark copy A"),
        ("attention_like.fut", ["run16cute", "run32cute", "run64cute", "run128cute"], "cudatc", "CUDA backend w/ TC, mixed f16/f32 & CuTe copy A"),
        ("attention_like.fut", ["run16futhark", "run32futhark", "run64futhark"], "cudatc", "CUDA backend w/ TC, mixed f16/f32 & Futhark copy A")
    ]
    to_plot = assemble(experiment_list, blocks, attention_result)

    plot_graph("Flash Attention Like, matrix multiplications of size $n\\times n \\times n$", "$n$", "TFLOPS", to_plot)

def lud_result(results, experiment, blocks):
    pass

def lud():
    blocks = 256*256
    experiment_list = [
        #("attention_like_orig.fut", "run_no_intra_f16", "cuda", "CUDA backend f16 w/ Futhark copy A"),
        #("attention_like_orig.fut", "run_no_intra_f32", "cuda", "CUDA backend f32 w/ Futhark copy A"),
        #("attention_like.fut", "run_f16", "cudatc", "CUDA backend w/ TC, mixed f16/f32 & Futhark copy A")
    ]
    to_plot = assemble(experiment_list, blocks, attention_result)

    plot_graph("Flash Attention Like, matrix multiplications of size $n\\times n \\times n$", "$n$", "TFLOPS", to_plot)

batched_mmm()
custom_attention()
attention()
#lud()
