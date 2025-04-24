#!/usr/bin/env nextflow

params.k_values = [1, 2]          // You can extend to [1, 2, 3]
params.gamma_values = [0]      // You can extend to [0, 0.25, 0.5, 0.75, 1.0]

workflow {
    k_gamma_ch = Channel
        .from(params.k_values)
        .combine(params.gamma_values)
        .map { k, gamma -> [k, gamma] }

    circulatory_script = file("${baseDir}/circulatory_flow_model.py")
    config_file = file("${baseDir}/figure2_config.ini")
    simulate(k_gamma_ch, circulatory_script, config_file)
        .collect()
        .set { sim_results }

    // fig2_plot_script = file("${baseDir}/plot_figure2.py")
    // plot(fig2_plot_script, sim_results)
    utils_script = file("${baseDir}/utils.py")
    jupyter_script = file("${baseDir}/plot_figure2.ipynb")
    generate_plot(sim_results, jupyter_script, utils_script)
}


process simulate {
    tag "${k}_${gamma}"

    input:
    tuple val(k), val(gamma)
    path 'circulatory_flow_model.py'
    path 'figure2_config.ini'

    output:
    path("flow_results/K${k}_gamma_${gamma}", emit: result_dir)

    script:
    """
    echo "...running gamma = ${gamma}, k = ${k}"
    python circulatory_flow_model.py \
        --config figure2_config.ini \
        --overrides General:outdir=flow_results/K${k}_gamma_${gamma} \
                    General:mixing_gamma=${gamma} \
                    Goods:hill_coefficient=${k} \
                    Toxins:hill_coefficient=${k}
    """
}


process generate_plot {
    input:
    path sim_dirs
    path 'plot_figure2.ipynb'
    path 'utils.py'
    // output:
    // path "figure2_plot.png"

    script:
    
    """
    echo "${sim_dirs}" > sim_paths.txt
    jupyter nbconvert --to notebook --execute plot_figure2.ipynb \\
        --ExecutePreprocessor.timeout=600 \\
        --output executed_plot_fig2.ipynb \\
        --ExecutePreprocessor.kernel_name=aging
    """
}
// do we need an output for this process?; could use to direct into another process that saves the final result in a given output directory, but do we need/want to do this?
// understand pathing better; including utils.py as input was enough, didn't need to copy over to 'here' ...?
// test with creating conda env in nf script with yml file instead...