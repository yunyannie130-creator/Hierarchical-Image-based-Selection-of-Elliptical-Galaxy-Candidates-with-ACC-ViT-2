# Catalog column descriptions

Copied from the Column guide sheet of the released workbook. Blank cells in the catalog indicate unavailable values, not zeros.

## beyond_hart16_com_candidates.xlsx

| Column | Description | Unit / source |
|---|---|---|
| sdss_bestobjid | SDSS BEST object identifier; stored as text to preserve all digits. | SDSS/Vavilova catalog |
| ra_deg | Right ascension in decimal degrees. | deg; Vavilova catalog |
| dec_deg | Declination in decimal degrees. | deg; Vavilova catalog |
| redshift | Catalog redshift. | Vavilova catalog |
| r_mag | Apparent r-band magnitude. | mag; Vavilova catalog |
| r_band_extinction_mag | Adopted r-band Galactic-extinction correction. | mag; Vavilova catalog |
| r_band_k_correction_mag | Adopted r-band K-correction. | mag; Vavilova catalog |
| absolute_r_mag | Absolute r-band magnitude reported by the source catalog. | mag; Vavilova catalog |
| r50_arcsec | Petrosian radius enclosing 50% of the light. | arcsec; Vavilova catalog |
| r90_arcsec | Petrosian radius enclosing 90% of the light. | arcsec; Vavilova catalog |
| luminosity_distance_mpc | Luminosity distance reported by the source catalog. | Mpc; Vavilova catalog |
| predicted_class | Final class assigned by the fixed ACC-ViT-2 pipeline. | This work |
| stage1_ell_probability | Stage-1 probability assigned to the ELL branch. | This work |
| stage2_com_probability | Stage-2 COM probability conditional on the ELL branch. | This work |
| joint_confidence | Minimum of the stage-1 ELL and stage-2 COM probabilities. | This work |
| confidence_ge_0p90 | True when both model probabilities are at least 0.90. | This work |
| confidence_ge_0p95 | True when both model probabilities are at least 0.95. | This work |
| apparent_size_bin | Apparent-size group: Small (R90 < 4), Medium (4 <= R90 < 6), or Large (R90 >= 6). | This work; R90 in arcsec |
| vavilova_completely_round_probability | Completely-round morphology probability in the comparison catalog. | Vavilova catalog |
| vavilova_in_between_probability | In-between morphology probability in the comparison catalog. | Vavilova catalog |
| vavilova_cigar_shaped_probability | Cigar-shaped morphology probability in the comparison catalog. | Vavilova catalog |
| vavilova_edge_on_probability | Edge-on morphology probability in the comparison catalog. | Vavilova catalog |
| vavilova_spiral_probability | Spiral morphology probability in the comparison catalog. | Vavilova catalog |
