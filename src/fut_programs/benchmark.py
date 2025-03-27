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
    #plt.xscale('log')
    filename = title[:title.index(',')].replace(' ', '_') if ',' in title else title.replace(' ', '_')
    plt.savefig(os.path.join("graphs", f"{filename}.pdf"))

def run_command(backend, results_file, entry_point, script, working_dir, tuning):
    args = [
        "futmma", 
        "bench", 
        f"--backend={backend}",
        "--pass-option=--nvrtc-option=-I../../../cutlass/include" if backend=="cudatc" else "",
        "--pass-option=-default-tile-size=16",
        "--pass-option=-default-reg-tile-size=4",
        f"--json={results_file}", 
        f"--tuning={tuning}" if tuning else "",
        f"--entry-point={entry_point}", 
        script
    ]
    args = [a for a in args if a != ""]
    subprocess.run(args, cwd=working_dir)

def assemble(experiment_list, blocks, result_func, n_lookups, working_dir):
    to_plot = []
    results_file = "tmp.json"
    results_path = os.path.join(working_dir, results_file)
    for list_entries in experiment_list:
        if len(list_entries) == 4:
            script, entry_point, backend, title = list_entries
            tuning = None
        elif len(list_entries) == 5:
            script, entry_point, backend, title, tuning = list_entries
        if type(entry_point) == list:
            tflops = [None]*len(entry_point)
            for i, ep in enumerate(entry_point):
                if type(tuning) == list:
                    t = tuning[i]
                else:
                    t = tuning
                    
                run_command(backend, results_file, ep, script, working_dir, t)

                with open(results_path, 'r') as json_file:
                    data = json.load(json_file)

                for ep, datasets in data.items():
                    for experiment, results in datasets["datasets"].items():
                        res, n = result_func(results, experiment, blocks)
                        if (res != -1):
                            tflops[n_lookups[n]] = res
            ns = [i for i in n_lookups.keys() if n_lookups[i] < len(tflops)]
            to_plot.append([ns, tflops, title])

        else:
            run_command(backend, results_file, entry_point, script, working_dir, tuning)

            with open(results_path, 'r') as json_file:
                data = json.load(json_file)

            for entry_point, datasets in data.items():
                tflops = [None]*len(n_lookups)
                for experiment, results in datasets["datasets"].items():
                    res, n = result_func(results, experiment, blocks)
                    if (res != -1):
                        tflops[n_lookups[n]] = res
                to_plot.append([list(n_lookups.keys()), tflops, title])
    os.remove(results_path)
    return to_plot
    
def batched_mmm_result(results, experiment, blocks):
    runtimes = results["runtimes"]
    mean_runtime = mean(runtimes)
    n = int(experiment[len(f"[{blocks}]["):nth(experiment, '[', 2)-1])
    tflops = (blocks * (n ** 3) * 2) / (mean_runtime * 1e6)
    return (tflops, n)

def batched_mmm():
    blocks = 32768
    working_dir = os.path.abspath("./batched-mmm")
    experiment_list = [
        ("batched_mmm_orig.fut", "mmmf16", "cuda", "CUDA backend f16"),
        ("batched_mmm_orig.fut", "mmmf32", "cuda", "CUDA backend f32"),
        ("batched_mmm.fut", ["mmm_intra16", "mmm_intra32", "mmm_intra64", "mmm_intra128"], "cudatc", "CUDA backend w/ TC & mixed f16/f32")
    ]
    n_lookups = {
        16: 0,
        32: 1,
        64: 2,
        128: 3
    }
    to_plot = assemble(experiment_list, blocks, batched_mmm_result, n_lookups, working_dir)

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
    working_dir = os.path.abspath("./custom-attention-like")
    n_lookups = {
        16: 0,
        32: 1,
        64: 2,
        128: 3
    }
    to_plot = assemble(experiment_list, blocks, custom_attention_result, n_lookups, working_dir)

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
    working_dir = os.path.abspath("./attention-like")
    n_lookups = {
        16: 0,
        32: 1,
        64: 2,
        128: 3
    }
    to_plot = assemble(experiment_list, blocks, attention_result, n_lookups, working_dir)

    plot_graph("Flash Attention Like, matrix multiplications of size $n\\times n \\times n$", "$n$", "TFLOPS", to_plot)

def lud_result(results, experiment, blocks):
    runtimes = results["runtimes"]
    mean_runtime = mean(runtimes)
    n = int(experiment[len(f"[256]["):nth(experiment, '[', 2)-1])
    tflops = (blocks * (n * n * n * 2 + n * n )) / (mean_runtime * 1e6)
    return (tflops, n)

def lud():
    blocks = 256*256
    experiment_list = [
        ("lud-mmm-orig.fut", "ludf16", "cuda", "CUDA backend f16"),
        ("lud-mmm-orig.fut", "ludf32", "cuda", "CUDA backend f32"),
        ("lud-mmm.fut", ["lud16","lud32","lud64","lud128"], "cudatc", "CUDA backend w/ TC, mixed f16/f32")
    ]
    working_dir = os.path.abspath("./lud-mmm")
    n_lookups = {
        16: 0,
        32: 1,
        64: 2,
        128: 3
    }
    to_plot = assemble(experiment_list, blocks, lud_result, n_lookups, working_dir)

    plot_graph("LUD like, matrix multiplications of size $n\\times n \\times n$", "$n$", "TFLOPS", to_plot)

def flash_full_result(results, experiment, blocks):
    runtimes = results["runtimes"]
    mean_runtime = mean(runtimes)
    if experiment.startswith("Class "):
        n = int(experiment[6:experiment.index('-')])
        d = int(experiment[experiment.index('-')+1:])
        tflops = (4 * d * n * n) / (mean_runtime * 1e6)
        return (tflops, d)
    elif experiment.startswith("Block "):
        blocks = int(experiment[6:experiment.index('-')])
        d = int(experiment[experiment.index('-')+1:])
        tflops = blocks * (d ** 3) * 2 * 2 / (mean_runtime * 1e6)
        return (tflops, d)
    else:
        return (-1, -1)

def flash_full():
    blocks = 1
    experiment_list = [
        #("flash-cfal-orig.fut", "main64", "cuda", "CUDA, d:64"),
        #("flash-cfal-orig.fut", "main128", "cuda", "CUDA, d:128"),
        #("flash-cfal-modified.fut", "main64", "cuda", "CUDA my backend w/ TC, d:64"),
        #("flash-cfal-modified.fut", "main128", "cuda", "CUDA my backend w/ TC, d:128"),
        #("flash-cfal-thesis.fut", "main64", "cuda", "CUDA thesis backend w/ TC, d:64"),
        #("flash-cfal-thesis.fut", "main128", "cuda", "CUDA thesis backend w/ TC, d:128"),
        ("flash-cfal-orig.fut", "thesislike16", "cuda", "basic CUDA f16", "tuning"),
        ("flash-cfal-orig.fut", "thesislike32", "cuda", "basic CUDA f32", "tuning"),
        #("flash-cfal-thesis.fut", ["thesislike16", "thesislike32", "thesislike64", "thesislike128"], "cudatc", "CUDA thesis backend w/ TC", ["tuning16", "tuning32", "tuning64", "tuning128"]),
        ("flash-cfal-modified.fut", ["thesislike16", "thesislike32", "thesislike64", "thesislike128", "thesislike256", "thesislike512"], "cudatc", "CUDA my backend w/ TC"),#, "xxx"),
    ]
    
    working_dir = os.path.abspath("./flash-full")
    d_lookups = {
        16: 0,
        32: 1,
        64: 2,
        128: 3,
        256: 4,
        512: 5
    }
    to_plot = assemble(experiment_list, blocks, flash_full_result, d_lookups, working_dir)

    plot_graph("Flash Attention, matrix size $m\\times n \\times n$, $m = 8192 / n$", "$n$", "TFLOPS", to_plot)

def large_mmm_result(results, experiment, blocks):
    runtimes = results["runtimes"]
    mean_runtime = mean(runtimes)
    first = experiment.split(' ')[0]
    if first.count('[') == 4:
        n = int(experiment[1:experiment.index(']')]) * 128
    else:
    #n = int(experiment[len(f"[{blocks}]["):nth(experiment, '[', 2)-1])
        n = int(experiment[1:experiment.index(']')])
    tflops = ((n ** 3) * 2) / (mean_runtime * 1e6)
    return (tflops, n)

def large_mmm():
    blocks = 32768

    experiment_list = [
        ("large-mmm-basic.fut", "mmm_f16", "cuda", "basic CUDA backend f16"),
        ("large-mmm-basic.fut", "mmm_f32", "cuda", "basic CUDA backend f32"),
        ("large-mmm-red-orig.fut", "main", "cuda", "CUDA backend?"),
        ("large-mmm-red.fut", ["run_square_small", "run_square_medium", "run_square_large", "run_square_xl"], "cudatc", "CUDA backend w/ TC & mixed f16/f32")
    ]
    working_dir = os.path.abspath("./large-mmm")
    n_lookups = {
        1024: 0,
        2048: 1,
        4096: 2,
        8192: 3
    }
    to_plot = assemble(experiment_list, blocks, large_mmm_result, n_lookups, working_dir)

    plot_graph("Large Matrix Multiplication, of size $n\\times n \\times n$", "$n$", "TFLOPS", to_plot)


#batched_mmm()
#custom_attention()
#attention()##TBD
#lud()
flash_full()
#large_mmm()