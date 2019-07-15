# PhD-Code
Hello, this repository contains `python` code for reproducing some of the examples presented in my Ph.D. manuscript

All the examples are carried out using datasets obtained via the MOABB framework, so the user is expected to have downloaded and installed the MOABB package (https://github.com/NeuroTechX/moabb)

For the moment (15th July 2019), it contains three scripts with examples :

- **Example 1** (*./examples/dimensionality-reduction/example1.py*)

  In this example, we compare the results of clasification with MDM when reducing the dimensionality of EEG recordings from the Physionet   MI dataset. We use the 'covpca' and 'gpcaRiemann' methods, as well as the SELg and SELb electrode selections. The user is referred to     Chapter 3 on my manuscript for more details.

- **Example 2** (*./examples/transfer-learning/example1.py*)

  In this example, we use Riemannian Procrustes analysis (RPA) to match the statistics of data points from two subjects in the Cho2017       database. We use a cross-validation routine to investigate the cross-subject classification scores. The user is referred to Chapter 4 on   my manuscript for more details.
  
- **Example 3** (*./examples/dimensionality-mismatch/example1.py*)

  In this example, we use the dimensionality transcending (DT) procedure to match the dimensionalities of two dataset: Zhou2016 (where       there are 14 electrodes) and BNCI2015001 (where there are 13 electrodes). The electrodes on the two datasets are positioned in different   places as well, so in the end with need to augment the dimensionalities of both datasets to 18. We use the same cross-validation           procedure for transfer learning to assess the quality of the classification using DT. The user is referred to Chapter 5 on my manuscript   for more details.

Cheers,

Pedro
