#!/usr/bin/env nextflow

params.velocity = [10]
params.diffusion = [20]
params.gamma = [0, 1]

workflow {
    velocity_gamma_ch = Channel
        .from(params.velocity)
        .combine(params.gamma)
        .map { velocity, gamma -> [velocity, gamma] }

    diffusion_gamma_ch = Channel
        .from(params.diffusion)
        .combine(params.gamma)
        .map { diffusion, gamma -> [diffusion, gamma] }

    circulatory_script = file("${baseDir}/circulatory_flow_model.py")
    config_file = file("${baseDir}/figure3_config.ini")
    simulate_vs_velocity(velocity_gamma_ch, circulatory_script, config_file)
        .collect()
        .set { vel_sim_results }

    simulate_vs_diffusion(diffusion_gamma_ch, circulatory_script, config_file)
        .collect()
        .set { diff_sim_results }

    utils_script = file("${baseDir}/utils.py")
    jupyter_script = file("${baseDir}/plot_figure3.ipynb")
    generate_plot(vel_sim_results, diff_sim_results, jupyter_script, utils_script)
}


process simulate_vs_velocity {
    tag "${velocity}_${gamma}"

    input:
    tuple val(velocity), val(gamma)
    path 'circulatory_flow_model.py'
    path 'figure3_config.ini'

    output:
    path("fig3_results/velocity_${velocity}_gamma_${gamma}", emit: result_dir)
    publishDir 'results/', mode: 'copy', pattern: "fig3_results/*"

    script:
    """
    echo "...running velocity = ${velocity} gamma = ${gamma}"
    python circulatory_flow_model.py \\
        --config figure3_config.ini \\
        --overrides General:outdir=fig3_results/velocity_${velocity}_gamma_${gamma} \\
                    General:flow_velocity=${velocity} \\
                    General:mixing_gamma=${gamma}
    """
}

process simulate_vs_diffusion {
    tag "${diffusion}_${gamma}"

    input:
    tuple val(diffusion), val(gamma)
    path 'circulatory_flow_model.py'
    path 'figure3_config.ini'

    output:
    path("fig3_results/diffusion_${diffusion}_gamma_${gamma}", emit: result_dir)
    publishDir 'results/', mode: 'copy', pattern: "fig3_results/*"

    script:
    """
    echo "...running diffusion = ${diffusion}, gamma = ${gamma}"
    python circulatory_flow_model.py \\
        --config figure3_config.ini \\
        --overrides General:outdir=fig3_results/diffusion_${diffusion}_gamma_${gamma} \\
                    General:diffusion_coefficient=${diffusion} \\
                    General:mixing_gamma=${gamma}
    """
}   

process generate_plot {
    input:
    path vel_sim_dirs
    path diff_sim_dirs
    path 'plot_figure3.ipynb'
    path 'utils.py'

    output:
    file '*'
    publishDir 'results/', mode: 'copy'

    script:
    """
    echo "...generating plot"
    echo "${vel_sim_dirs}" > vel_sim_paths.txt
    echo "${diff_sim_dirs}" > diff_sim_paths.txt
    jupyter nbconvert --to notebook --execute plot_figure3.ipynb \\
        --ExecutePreprocessor.timeout=600 \\
        --output executed_plot_fig3.ipynb \\
        --ExecutePreprocessor.kernel_name=aging
    """
}