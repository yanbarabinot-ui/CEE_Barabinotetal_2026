# CEE_Barabinotetal_2026
python codes

compare_ke_l3_duacs_l4_swot_like.py --> build DUACS EKE maps "swot-like" + build csv file for the EKE time series 

derive_spectra_pdf_l3.py --> SSH spectrum + PDFs for SWOT L3 data + maps of strain and relative vorticity (figure 5 in the article)

fetch_swot_l3_expert_cycle.py --> download one chosen cycle of SWOT from AVISO website 

fetch_swot_l3_expert_multi_cycle.py --> run fetch_swot_l3_expert_cycle.py to download SWOT cycles you want 

run_cycles_swot_topo.py --> construct the csv files with mean EKE, STD and standard error per cycles (useful for plot_figure_3.py)

eke_mean_std_over_cycles.py --> build EKE maps from SWOT, figure 1 and 2 in the article

compute_med_ke_binned.py --> useful for eke_mean_std_over_cycles.py, figure 1 and 2 in the article

seasonal_eke_diff_swot_duacs --> to plot figure 4 in the article 

seasonal_joint_pdfs_duacs.py --> joint pdf for DUACS, figure 6 in the article

seasonal_joint_pdfs.py --> joint pdf for SWOT, figure 6 in the article

plot_figure_3.py --> to plot figure 3 in the article 

plot_figures_spectra_pdf.py --> to plot spectra or single pdf (not joint) for several products (DUACS, SWOT)

plot_figure_7.py --> to plot figure 7 in the article

