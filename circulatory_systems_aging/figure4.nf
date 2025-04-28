#!/usr/bin/env nextflow

nextflow.enable.dsl=2
// add CUDA_LAUNCH_BLOCKING=1


params.runtime = [1] // [40 40 40 40 40 40 10 10 10 10 10]
params.gamma = [1.0] // [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
params.k_values = [1,2,3]

workflow {
    // input_ch = Channel
    //     .from(params.run)

    runtime_ch = Channel
        .from(params.runtime)
    gamma_ch = Channel
        .from(params.gamma)
    runtime_gamma_ch = runtime_ch
        .merge(gamma_ch)
    k_ch = Channel
        .from(params.k_values)
    combined_ch = runtime_gamma_ch
        .combine(k_ch)
        .map { runtime, gamma, k -> [runtime, gamma, k] }
        
    circulatory_script = file("${baseDir}/circulatory_flow_model.py")
    config_file = file("${baseDir}/figure4_config.ini")
    run_base_sims(combined_ch, circulatory_script, config_file)
        .collect()
        .set { sim_results }
}

process run_base_sims {
    maxForks 1

    tag "${runtime}_${gamma}_${k}"

    input:
    tuple val(runtime), val(gamma), val(k)
    path 'circulatory_flow_model.py'
    path 'figure4_config.ini'

    output:
    path("fig4_results/runtime_${runtime}_gamma_${gamma}_k_${k}", emit: result_dir)
    publishDir 'results/', mode: 'copy', pattern: "fig4_results/*"

    script:
    """
    echo "...running runtime = ${runtime} gamma = ${gamma} k = ${k}"
    python circulatory_flow_model.py \\
        --config figure4_config.ini \\
        --overrides General:outdir=fig4_results/runtime_${runtime}_gamma_${gamma}_k_${k} \\
                    General:mixing_gamma=${gamma} \\
                    Goods:hill_coefficient=${k} \\
                    Toxins:hill_coefficient=${k} \\
                    General:runtime=${runtime}
    """
}
