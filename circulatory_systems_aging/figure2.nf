#!/usr/bin/env nextflow

params.k_values = [1]          // You can extend to [1, 2, 3]
params.gamma_values = [0]      // You can extend to [0, 0.25, 0.5, 0.75, 1.0]

workflow {
    k_gamma_ch = Channel
        .from(params.k_values)
        .combine(params.gamma_values)
        .map { k, gamma -> [k, gamma] }

    circultory_script = file("${baseDir}/ccirculatory_flow_model.py")
    config_file = file("${baseDir}/figure2_config.ini")
    simulate(k_gamma_ch, circultory_script, config_file)
        .collect()
        .set { sim_results }

    // plot(sim_results)
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

// TODO: figure out how to save results to a results directory, not in work directory...