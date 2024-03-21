import os
from os.path import dirname
import pandas as pd
import json

from ..utils import DatasetUtils


class LoadDataset:

    @staticmethod
    def load_bbcsport_dataset(p = 0, return_y: bool = False, shuffle: bool = True,
                              assess_percentage: bool = True, random_state: int = None):
        r"""
        The BBCSport dataset comprises five kinds of sports news articles (i.e., athletics, cricket, football, rugby,
        and tennis) collected from the BBC Sport website. This specific subset includes 116 samples and is 
        represented by four different views.

        Parameters
        ----------
        p: list or float
            The percentaje that each view will have for missing samples. If p is float, all the views will have the
            same percentaje.
        return_y: bool, default False
            If True, return the label too.
        shuffle: bool, default False
            If True, shuffle the dataset.
        assess_percentage: bool
            If False, each view is dropped independently.
        random_state: int, default None
            If int, random_state is the seed used by the random number generator.

        Returns
        -------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        y : optional
            Array with labels

        References
        ----------
        [#1paper] D. Greene and P. Cunningham, “Practical solutions to the problem of diagonal dominance in kernel 
        document clustering,” in ICML. ACM, 2006, pp. 377–384.
        [#2paper] N. Rai, S. Negi, S. Chaudhury, and O. Deshmukh, “Partial multi-view clustering using graph 
        regularized nmf,” in ICPR. IEEE, 2016, pp.2192–2197.
        [#1url] https://github.com/GPMVCDummy/GPMVC/tree/master/partialMV/PVC/recreateResults/data

         Examples
        --------
        >>> from imvc.datasets import LoadDataset
        >>> Xs = LoadDataset.load_bbcsport_dataset(p = 0.2)
        """
        output = LoadDataset.load_dataset(dataset_name= "bbcsport", p=p, return_y = return_y,
                               shuffle = shuffle, assess_percentage = assess_percentage, random_state = random_state)
        return output

    
    @staticmethod
    def load_bdgp_dataset(p = 0, return_y: bool = False, return_metadata: bool = False, shuffle: bool = True, 
                          assess_percentage: bool = True, random_state: int = None):
        r"""
        The BDGP (Berkeley Drosophila Genome Project) is an image dataset focused on drosophila embryos, 
        containing 2,500 samples featuring five distinct objects. Each sample within this dataset is characterized
        by a 1750-dimensional visual feature and a 79-dimensional textual feature.

        Parameters
        ----------
        p: list or float
            The percentaje that each view will have for missing samples. If p is float, all the views will have the
            same percentaje.
        return_y: bool, default False
            If True, return the label too.
        return_metadata: bool, default False
            If True, return the metadata.
        shuffle: bool, default False
            If True, shuffle the dataset.
        assess_percentage: bool
            If False, each view is dropped independently.
        random_state: int, default None
            If int, random_state is the seed used by the random number generator.

        Returns
        -------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        y : optional
            Array with labels
        metadata : optional
            Dict with info about the dataset (data modality names, labels, etc.).

        References
        ----------
        [#1paper] X. Cai, H. Wang, H. Huang, and C. Ding, “Joint stage recognition and anatomical annotation of 
        drosophila gene expression patterns,” Bioinformatics, vol. 28, no. 12, pp. i16–i24, 2012.

         Examples
        --------
        >>> from imvc.datasets import LoadDataset
        >>> Xs = LoadDataset.load_bdgp_dataset(p = 0.2)
        """
        output = LoadDataset.load_dataset(dataset_name= "bdgp", p=p, return_y = return_y, return_metadata = return_metadata,
                                       shuffle = shuffle, assess_percentage = assess_percentage, random_state = random_state)
        return output

    
    @staticmethod
    def load_caltech101_dataset(p = 0, return_y: bool = False, return_metadata: bool = False, shuffle: bool = True,
                              assess_percentage: bool = True, random_state: int = None):
        r"""
        The Caltech101 dataset is a widely used collection of objects, comprising 9,144 images spread across 102 
        categories, which include a background category and 101 distinct objects like airplanes, ants, bass, and
        beavers. Four types of feature sets have been selected, namely Cenhist, Hog, Gist, and LBP.

        Parameters
        ----------
        p: list or float
            The percentaje that each view will have for missing samples. If p is float, all the views will have the
            same percentaje.
        return_y: bool, default False
            If True, return the label too.
        return_metadata: bool, default False
            If True, return the metadata.
        shuffle: bool, default False
            If True, shuffle the dataset.
        assess_percentage: bool
            If False, each view is dropped independently.
        random_state: int, default None
            If int, random_state is the seed used by the random number generator.

        Returns
        -------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        y : optional
            Array with labels
        metadata : optional
            Dict with info about the dataset (data modality names, labels, etc.).

        References
        ----------
        [#1paper] L. Fei-Fei, R. Fergus, and P. Perona, “Learning generative visual models from few training 
        examples: An incremental bayesian approach tested on 101 object categories,” in CVPR Workshop. IEEE,
        2004, pp. 178–178.
        [#2paper] Li, F. Nie, H. Huang, and J. Huang, “Large-scale multi-view spectral clustering via bipartite
        graph,” in AAAI, 2015, pp. 2750–2756.
        [#1url] https://drive.google.com/drive/folders/1O3YmthAZGiq1ZPSdE74R7Nwos2PmnHH

         Examples
        --------
        >>> from imvc.datasets import LoadDataset
        >>> Xs = LoadDataset.load_caltech101_dataset(p = 0.2)
        """
        output = LoadDataset.load_dataset(dataset_name= "caltech101", p=p, return_y = return_y, return_metadata = return_metadata,
                               shuffle = shuffle, assess_percentage = assess_percentage, random_state = random_state)
        return output

    
    @staticmethod
    def load_digits_dataset(p = 0, return_y: bool = False, return_metadata: bool = False, shuffle: bool = True,
                              assess_percentage: bool = True, random_state: int = None):
        r"""
        This UCI multiple dataset showcases handwritten digit images categorized into classes 0-9, encompassing six distinct views.
        Each class comprises 200 labeled examples.

        Parameters
        ----------
        p: list or float
            The percentaje that each view will have for missing samples. If p is float, all the views will have the
            same percentaje.
        return_y: bool, default False
            If True, return the label too.
        return_metadata: bool, default False
            If True, return the metadata.
        shuffle: bool, default False
            If True, shuffle the dataset.
        assess_percentage: bool
            If False, each view is dropped independently.
        random_state: int, default None
            If int, random_state is the seed used by the random number generator.

        Returns
        -------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        y : optional
            Array with labels
        metadata : optional
            Dict with info about the dataset (data modality names, labels, etc.).

        References
        ----------
        [#1paper] M. van Breukelen, et al. "Handwritten digit recognition by combined classifiers", Kybernetika, 34(4):381-386,
        1998.
        [#2paper] Perry, Ronan, et al. "mvlearn: Multiview Machine Learning in Python." Journal of Machine Learning Research
        22.109 (2021): 1-7.
        [#1url] Dheeru Dua and Casey Graff. UCI machine learning repository, 2017. URL http://archive.ics.uci.edu/ml.

         Examples
        --------
        >>> from imvc.datasets import LoadDataset
        >>> Xs = LoadDataset.load_digits_dataset(p = 0.2)
        """
        output = LoadDataset.load_dataset(dataset_name= "bbcsport", p=p, return_y = return_y, return_metadata = return_metadata,
                               shuffle = shuffle, assess_percentage = assess_percentage, random_state = random_state)
        return output

    
    @staticmethod
    def load_nuswide_dataset(p = 0, return_y: bool = False, shuffle: bool = True,
                              assess_percentage: bool = True, random_state: int = None):
        r"""
        The NUSWIDE dataset, created by researchers from the National University of Singapore, is a collection of real-world
        web images. This is subset that includes 30,000 images across 31 distinct classes. Each image in this subset is 
        represented by a combination of five types of low-level features, namely: color histogram; color correlogram; edge 
        direction histogram; wavelet texture and block-wise color moments.

        Parameters
        ----------
        p: list or float
            The percentaje that each view will have for missing samples. If p is float, all the views will have the
            same percentaje.
        return_y: bool, default False
            If True, return the label too.
        shuffle: bool, default False
            If True, shuffle the dataset.
        assess_percentage: bool
            If False, each view is dropped independently.
        random_state: int, default None
            If int, random_state is the seed used by the random number generator.

        Returns
        -------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        y : optional
            Array with labels

        References
        ----------
        [#1paper] T.-S. Chua, J. Tang, R. Hong, H. Li, Z. Luo, and Y. Zheng, “Nus-wide: a real-world web image database from
        national university of singapore,” in ACM ICIVR, 2009, pp. 1–9.
        [#1url] https://drive.google.com/drive/folders/1O3YmthAZGiq1ZPSdE74R7Nwos2PmnHH

         Examples
        --------
        >>> from imvc.datasets import LoadDataset
        >>> Xs = LoadDataset.load_nuswide_dataset(p = 0.2)
        """
        output = LoadDataset.load_dataset(dataset_name= "nuswide", p=p, return_y = return_y,
                               shuffle = shuffle, assess_percentage = assess_percentage, random_state = random_state)
        return output

    
    def load_nutrimouse_dataset(p = 0, return_y: bool = False, return_metadata: bool = False, shuffle: bool = True,
                              assess_percentage: bool = True, random_state: int = None):
        r"""
        The nutrimouse dataset originates from a mouse nutrition study conducted by Pascal Martin at the Toxicology
        and Pharmacology Laboratory within the French National Institute for Agronomic Research. This dataset encompasses 
        information from 40 mice, offering two distinct data modalities: expression levels of potentially relevant genes
        (comprising 120 numerical values) and concentrations of specific fatty acids (involving 21 numerical variables).
        Each mouse within this dataset is characterized by two labels: its genetic type, divided into two classes, and 
        its diet, categorized into five classes.

        Parameters
        ----------
        p: list or float
            The percentaje that each view will have for missing samples. If p is float, all the views will have the
            same percentaje.
        return_y: bool, default False
            If True, return the label too.
        return_metadata: bool, default False
            If True, return the metadata.
        shuffle: bool, default False
            If True, shuffle the dataset.
        assess_percentage: bool
            If False, each view is dropped independently.
        random_state: int, default None
            If int, random_state is the seed used by the random number generator.

        Returns
        -------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        y : optional
            Array with labels
        metadata : optional
            Dict with info about the dataset (data modality names, labels, etc.).

        References
        ----------
        [#1paper] P. Martin, H. Guillou, F. Lasserre, S. Déjean, A. Lan, J-M. Pascussi, M. San Cristobal, P. Legrand,
        P. Besse, T. Pineau. "Novel aspects of PPARalpha-mediated regulation of lipid and xenobiotic metabolism revealed 
        through a nutrigenomic study." Hepatology, 2007.
        [#2paper] González I., Déjean S., Martin P.G.P and Baccini, A. (2008) CCA: "An R Package to Extend Canonical
        Correlation Analysis." Journal of Statistical Software, 23(12).
        [#3paper] Perry, Ronan, et al. "mvlearn: Multiview Machine Learning in Python." Journal of Machine Learning 
        Research 22.109 (2021): 1-7.

         Examples
        --------
        >>> from imvc.datasets import LoadDataset
        >>> Xs = LoadDataset.load_nutrimouse_dataset(p = 0.2)
        """
        output = LoadDataset.load_dataset(dataset_name= "nutrimouse", p=p, return_y = return_y, return_metadata = return_metadata,
                               shuffle = shuffle, assess_percentage = assess_percentage, random_state = random_state)
        return output

    
    @staticmethod
    def load_tcga_dataset(p = 0, return_y: bool = False, return_metadata: bool = False, shuffle: bool = True,
                              assess_percentage: bool = True, random_state: int = None):
        r"""
        This dataset is composed of ten cancer types multi-omics data from The Cancer Genome Atlas (TCGA). This is a subset
        composed of four kinds of data: mRNA, miRNA, DNA-methylation and proteomics. Two possible targets are provided:
        origin tissue (10 labels) and survival data (numerical values).

        Parameters
        ----------
        p: list or float
            The percentaje that each view will have for missing samples. If p is float, all the views will have the
            same percentaje.
        return_y: bool, default False
            If True, return the label too.
        return_metadata: bool, default False
            If True, return the metadata.
        shuffle: bool, default False
            If True, shuffle the dataset.
        assess_percentage: bool
            If False, each view is dropped independently.
        random_state: int, default None
            If int, random_state is the seed used by the random number generator.

        Returns
        -------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        y : optional
            Array with labels
        metadata : optional
            Dict with info about the dataset (data modality names, labels, etc.).

        References
        ----------
        [#1paper] Hoadley, Katherine & Yau, Christina & Wolf, Denise & Cherniack, Andrew & Tamborero, David & Ng, Sam & 
        Leiserson, Mark & Niu, Shubin & Mclellan, Michael & Uzunangelov, Vladislav & Zhang, Jiashan & Kandoth, Cyriac & 
        Akbani, Rehan & Shen, Hui & Omberg, Larsson & Chu, Andy & Margolin, Adam & van 't Veer, Laura & López-Bigas, Nuria
        & Zou, Lihua. (2014). Multiplatform Analysis of 12 Cancer Types Reveals Molecular Classification within and across
        Tissues of Origin. Cell. 158. 10.1016/j.cell.2014.06.049.
        [#1url] https://www.synapse.org

         Examples
        --------
        >>> from imvc.datasets import LoadDataset
        >>> Xs = LoadDataset.load_tcga_dataset(p = 0.2)
        """
        output = LoadDataset.load_dataset(dataset_name= "tcga", p=p, return_y = return_y, return_metadata = return_metadata,
                               shuffle = shuffle, assess_percentage = assess_percentage, random_state = random_state)
        return output

    
    @staticmethod
    def load_dataset(dataset_name: "str", return_y: bool = False, return_metadata: bool = False, shuffle: bool = True,
                     random_state: int = None):
        r"""
        Load a multi-modal dataset.

        Parameters
        ----------
        return_y: bool, default False
            If True, return the label too.
        return_metadata: bool, default False
            If True, return the metadata.
        shuffle: bool, default False
            If True, shuffle the dataset.
        random_state: int, default None
            If int, random_state is the seed used by the random number generator.

        Returns
        -------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.
        y : optional
            Array with labels
        metadata : optional
            Dict with info about the dataset (data modality names, labels, etc.).

         Examples
        --------
        >>> from imvc.datasets import LoadDataset
        >>> Xs = LoadDataset.load_dataset(dataset_name = 'tcga', p = 0.2)
        """
        module_path = dirname(__file__)
        data_path = os.path.join(module_path, "data", dataset_name)
        data_files = [filename for filename in os.listdir(data_path)]
        data_files = sorted(data_files)
        data_files = [os.path.join(data_path, filename) for filename in data_files if dataset_name in filename and not filename.endswith("y.csv")]
        Xs = [pd.read_csv(filename) for filename in data_files]
        # Xs = DatasetUtils.add_random_noise_to_views(Xs=Xs, p=p, assess_percentage=assess_percentage, random_state=random_state)
        if shuffle:
            Xs = DatasetUtils.shuffle_imvd(Xs=Xs, random_state=random_state)
        output = (Xs,)
        if return_y:
            y = pd.read_csv(os.path.join(data_path, f"{dataset_name}_y.csv"))
            y = y.loc[Xs[0].index]
            if y.shape[1] > 1:
                y = y.squeeze()
            output = output + (y,)
        if return_metadata:
            metadata_filename = os.path.join(data_path, "metadata.json")
            if os.path.isfile(metadata_filename):
                with open(metadata_filename) as json_file:
                    metadata = json.load(json_file)
                output = output + (metadata,)
            else:
                output = output + (None,)
        if len(output) == 1:
            output = output[0]
        return output
