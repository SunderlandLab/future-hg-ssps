#!/bin/bash

# -- fetch files from Harvard Dataverse
python fetch_dataverse_files.py

# -- this takes a long time to run
python attribute_regional_deposition.py

# -- display main results -- 
python display_primary_emissions_metrics.py

# -- plotting scripts --
python plot_1_regional_and_category_emissions.py
python plot_2_emission_speciation_trends.py
python plot_3_relative_regional_deposition_change.py
python plot_4_global_deposition_maps.py
python plot_5_legacy_deposition_trends.py
python plot_6_area_normalized_legacy_deposition.py
python plot_S1_compare_to_recent_emissions.py
python plot_S2_own_source_dep.py
python plot_S3_legacy_vs_primary_dep.py
python plot_S4_legacy_deposition_change_map.py
python plot_S5_upper_ocean_mass_trends.py
python plot_S6_source_receptor_matrices.py
