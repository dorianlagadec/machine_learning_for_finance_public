import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod


# %%%%%%%%%%%%% DataViewer %%%%%%%%%%%%%
class DataViewer:

    def __init__(
            self, 
            df:pd.DataFrame, 
            target:pd.DataFrame
    ):
        
        self.df = df
        self.y = target
        self.float_cols = self.df.select_dtypes(include=['float64']).columns
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns
    
    
    def stats_analysis(self)->pd.DataFrame:

        stats = self.df.describe(include='all').transpose()
        stats["dtype"] = self.df.dtypes
        stats["distinct_values"] = self.df.nunique()
        stats["count_nan"] = self.df.isna().sum()
        stats["nan_percentage"] = (self.df.isna().sum() / len(self.df)) * 100

        cols_order = ['dtype', 'count_nan', 'nan_percentage', 'distinct_values'] 
        remaining_cols = [c for c in stats.columns if c not in cols_order]

        final = stats[cols_order + remaining_cols]
        
        return final.drop(columns=['unique', 'top', 'freq'])
    

    def target_analysis(self):

        stats_target = self.y.describe().transpose()
        
        return stats_target
    

    def float_columns_normality_test(self)->pd.DataFrame:
        
        normality_results = {}

        for col in self.float_cols:
            stat, p_value = jarque_bera(self.df[col].dropna())
            normality_results[col] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }

        return pd.DataFrame(normality_results).transpose()
    

    def plot_float_distributions(self):

        fig, axes = plt.subplots(len(self.float_cols), 1, figsize=(10, 5 * len(self.float_cols)))
        if len(self.float_cols) == 1:
            axes = [axes]

        for i, col in enumerate(self.float_cols):
            sns.histplot(self.df[col].dropna(), kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()


    def plot_target_distribution(self):

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        sns.histplot(self.y['OS_YEARS'], kde=True, ax=axes[0], color='salmon')
        axes[0].set_title('Distribution life survival duration (OS_YEARS)')

        sns.countplot(x='OS_STATUS', data=self.y, ax=axes[1], palette='Set2')
        axes[1].set_title('Distribution of survival status (OS_STATUS)')

        plt.tight_layout()
        plt.show()


    def plot_correlation_matrix(self):

        plt.figure(figsize=(12, 8))

        corr = self.df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix of Numerical Features")
        plt.show()

    
    def plot_distribution_by_target(self):

        temp_df = self.df.merge(self.y[['OS_YEARS', 'OS_STATUS']], left_index=True, right_index=True)

        fig, axes = plt.subplots(len(self.df.columns), 1, figsize=(10, 5 * len(self.df.columns)))

        for i, col in enumerate(self.df.columns):
            # to many categories in CYTOGENETICS, skip for now
            if col=="CYTOGENETICS":
                continue
            sns.scatterplot(data=temp_df, x='OS_YEARS', y=col, hue='OS_STATUS', ax=axes[i], alpha=0.5)
            axes[i].set_title(f'{col} vs Years of Survival', fontsize=14)

        plt.tight_layout()
        plt.show()


    def plot_categorical_boxplot(self, categorical_col: list = None):

        temp_df = self.df.merge(
            self.y[['OS_YEARS', 'OS_STATUS']],
            left_index=True,
            right_index=True
        )

        cat_cols = self.categorical_cols if categorical_col is None else categorical_col

        for col in cat_cols:

            # to many categories in CYTOGENETICS, skip for now
            if col=="CYTOGENETICS":
                continue

            plt.figure(figsize=(max(12, len(temp_df[col].unique()) * 0.25), 6))
            temp_df = temp_df.reset_index(drop=True)
            sns.boxplot(
                data=temp_df,
                x=col,
                y='OS_YEARS'
            )

            plt.title(f'OS_YEARS by {col}', fontsize=14)
            plt.xlabel(col)
            plt.ylabel('OS_YEARS')

            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

    
    def plot_float_boxplot(self):

        temp_df = self.df.merge(self.y[['OS_YEARS', 'OS_STATUS']], left_index=True, right_index=True)

        fig, axes = plt.subplots(len(self.float_cols), 1, figsize=(10, 5 * len(self.float_cols)))

        for i, col in enumerate(self.float_cols):
            # to many categories in CYTOGENETICS, skip for now
            if col=="CYTOGENETICS":
                continue
            sns.boxplot(data=temp_df, x='OS_STATUS', y=col, ax=axes[i], palette='Set2')
            axes[i].set_title(f'{col} by Survival Status', fontsize=14)

        plt.tight_layout()
        plt.show()

    
    def corr_cytogenetics(self):
        #fonction qui calcule la corrélation entre le fait d'avoir un NaN dans la CYTOGENETICS et dans les autres variables (est-ce plus probable d'avoir un NaN dans les autres variables si on a un NaN dans la CYTOGENETICS ?)
        temp_df = self.df.copy()
        temp_df['CYTOGENETICS_is_nan'] = temp_df['CYTOGENETICS'].isna().astype(int)  
        corr_results = {}
        for col in self.df.columns:
            if col != 'CYTOGENETICS':
                temp_df[col+'_is_nan'] = temp_df[col].isna().astype(int)
                corr = temp_df['CYTOGENETICS_is_nan'].corr(temp_df[col+'_is_nan'])
                corr_results[col] = corr
        return pd.DataFrame.from_dict(corr_results, orient='index', columns=['Correlation_with_CYTOGENETICS_NaN'])
    



# %%%%%%%%%%%%% DataHandler %%%%%%%%%%%%%

# === Abstract Base Class for Data Handling ===
class DataHandler(ABC):

    def __init__(self):
        self.df = None
        self.y = None
        self.categorical_cols = []
        self.float_cols = []
        self.binary_cols= []

    @abstractmethod
    def aggregator(self):
        pass

    @abstractmethod
    def categorize(self):
        pass

    @abstractmethod
    def decode_cytogen(self):
        pass

    @abstractmethod
    def prepare(self):
        pass
#------------------------------


# === Building a Default Data Handler ===
class DefaultDataHandler(DataHandler):

    def __init__(self, clinical_df: pd.DataFrame, molecular_df:pd.DataFrame, target:pd.DataFrame):
        super().__init__()
        self.clinical_df = clinical_df
        self.molecular_df = molecular_df
        self.y = target


    def decode_cytogen(self):
        # Backwards-compatible mutating wrapper: operate on internal clinical_df
        self.clinical_df = self._decode_cytogen(self.clinical_df)
        

    def aggregator(self):
        # Backwards-compatible mutating wrapper: operate on internal dfs
        self.df = self._aggregator(self.clinical_df, self.molecular_df)
        

    def categorize(self):
        # Backwards-compatible mutating wrapper: set attributes based on current df
        cat, bin_cols, flt = self._categorize(self.df)
        self.categorical_cols = cat
        self.binary_cols = bin_cols
        self.float_cols = flt

    
    def drop_nan_target(self):
        """
        Drop rows where target contains NaN.
        Ensures perfect alignment between X and y.
        """
        # Backwards-compatible mutating wrapper: filter internal dfs
        df, y, molecular = self._drop_nan_target(self.df, self.y, self.molecular_df)
        self.df = df
        self.y = y
        self.molecular_df = molecular

    def prepare(self)->tuple[pd.DataFrame, pd.DataFrame, list, list, list]:
        """
        Prepare and return cleaned copies without mutating the handler's internal state.

        Returns: (df, y, float_cols, categorical_cols, binary_cols)
        """
        clinical = self.clinical_df.copy()
        molecular = self.molecular_df.copy() if self.molecular_df is not None else None
        y = self.y.copy() if self.y is not None else None

        clinical = self._decode_cytogen(clinical)
        df = self._aggregator(clinical, molecular)
        cat_cols, bin_cols, flt_cols = self._categorize(df)
        df, y, molecular = self._drop_nan_target(df, y, molecular)

        return (df, y, flt_cols, cat_cols, bin_cols)

    # ----- Private helpers on DataFrame copies -----
    def _decode_cytogen(self, clinical_df: pd.DataFrame) -> pd.DataFrame:
        cyto = clinical_df['CYTOGENETICS']

        cyto = (
            cyto.str.lower()
                .str.replace(r'\[.*?\]', '', regex=True)
                .str.replace('onfish', '', regex=False)
                .str.strip()
        )

        clinical_df['cyto_normal'] = cyto.str.fullmatch(r'46,(xx|xy)').astype("Int64").where(cyto.notna(), pd.NA)
        clinical_df['cyto_complex'] = cyto.str.contains('complex', na=False).astype("Int64").where(cyto.notna(), pd.NA)
        clinical_df['monosomy_7'] = cyto.str.contains(r'-7', na=False).astype("Int64").where(cyto.notna(), pd.NA)
        clinical_df['trisomy_8'] = cyto.str.contains(r'\+8', na=False).astype("Int64").where(cyto.notna(), pd.NA)
        clinical_df['del_5q'] = cyto.str.contains(r'del\(5', na=False).astype("Int64").where(cyto.notna(), pd.NA)
        clinical_df['t_3_3'] = cyto.str.contains(r't\(3;3\)', na=False).astype("Int64").where(cyto.notna(), pd.NA)
        clinical_df['n_abnormalities'] = cyto.str.count(r'del|add|dic|der|inv|t\(|\+|-').where(cyto.notna(), pd.NA)
        clinical_df['cyto_mosaic'] = cyto.str.contains('/', na=False).astype("Int64").where(cyto.notna(), pd.NA)

        clinical_df = clinical_df.drop('CYTOGENETICS', axis=1)
        return clinical_df

    def _aggregator(self, clinical_df: pd.DataFrame, molecular_df: pd.DataFrame) -> pd.DataFrame:
        if molecular_df is None:
            return clinical_df.copy()

        mol_agg = molecular_df.groupby("ID").agg(
            nb_mutations=("GENE", "count"),
            mean_vaf=("VAF", "mean"),
            max_vaf=("VAF", "max"),
        )

        # Ensure join is explicit on index
        if clinical_df.index.dtype == mol_agg.index.dtype or 'ID' in clinical_df.columns:
            # try to join on index; if IDs are a column, prefer merge
            try:
                joined = clinical_df.join(mol_agg)
            except Exception:
                joined = clinical_df.reset_index().merge(mol_agg.reset_index(), left_on='ID', right_on='ID', how='left').set_index(clinical_df.index.name)
        else:
            joined = clinical_df.join(mol_agg)

        return joined

    def _categorize(self, df: pd.DataFrame):
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        binary_cols = ['cyto_normal', 'cyto_complex', 'monosomy_7', 'trisomy_8', 'del_5q', 't_3_3', 'cyto_mosaic']
        float_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return categorical_cols, binary_cols, float_cols

    def _drop_nan_target(self, df: pd.DataFrame, y: pd.DataFrame, molecular_df: pd.DataFrame):
        if y is None:
            return df, y, molecular_df

        valid_idx = y.dropna().index
        y_filtered = y.loc[valid_idx]

        df_filtered = df.loc[valid_idx] if df is not None else None
        clinical_filtered = None
        molecular_filtered = None

        if molecular_df is not None:
            # molecular_df often contains rows per mutation; filter by ID column if present
            if 'ID' in molecular_df.columns:
                molecular_filtered = molecular_df[molecular_df['ID'].isin(valid_idx)].copy()
            else:
                molecular_filtered = molecular_df.loc[molecular_df.index.intersection(valid_idx)]

        return df_filtered, y_filtered, molecular_filtered


# === Building a Imporved Data Handler - with slitghtly different aggregator ===
class ImprovedDataHandler(DataHandler):

    def __init__(self, clinical_df: pd.DataFrame, molecular_df:pd.DataFrame, target:pd.DataFrame):
        super().__init__()
        self.clinical_df = clinical_df
        self.molecular_df = molecular_df
        self.y = target


    def decode_cytogen(self):
        self.clinical_df = self._decode_cytogen(self.clinical_df)
        

    def aggregator(self):
        self.df = self._aggregator(self.clinical_df, self.molecular_df)
        

    def categorize(self):
        cat, bin_cols, flt = self._categorize(self.df)
        self.categorical_cols = cat
        self.binary_cols = bin_cols
        self.float_cols = flt

    
    def drop_nan_target(self):
        """
        Drop rows where target contains NaN.
        Ensures perfect alignment between X and y.
        """
        # Backwards-compatible mutating wrapper: filter internal dfs
        df, y, molecular = self._drop_nan_target(self.df, self.y, self.molecular_df)
        self.df = df
        self.y = y
        self.molecular_df = molecular

    def prepare(self)->tuple[pd.DataFrame, pd.DataFrame, list, list, list]:
        """
        Prepare and return cleaned copies without mutating the handler's internal state.

        Returns: (df, y, float_cols, categorical_cols, binary_cols)
        """
        clinical = self.clinical_df.copy()
        molecular = self.molecular_df.copy() if self.molecular_df is not None else None
        y = self.y.copy() if self.y is not None else None

        clinical = self._decode_cytogen(clinical)
        molecular = self._decode_genes(molecular)
        df = self._aggregator(clinical, molecular)
        cat_cols, bin_cols, flt_cols = self._categorize(df)
        df, y, molecular = self._drop_nan_target(df, y, molecular)

        return (df, y, flt_cols, cat_cols, bin_cols)

    # ----- Private helpers on DataFrame copies -----
    def _decode_cytogen(self, clinical_df: pd.DataFrame) -> pd.DataFrame:
        cyto = clinical_df['CYTOGENETICS']

         # function that interprets the results of caryotipic analysis of the patients.
        # this has an impact on their prognosis
        # normal: don't have visible abnormalities
        # complex: genomic instability, treated more aggressively
        # monosomy 7: loss of a chromosome in the 7th pair, affecting genes involved in hematopoeiosis: poor prognosis
        # trisomy 8: extra copy of chromosome in the 8th pair
        # del 5q: loss of the long arm of chromosome 5, patients respond well to a treatment (lenalidomide)
        # t_3_3: translocation between chromosome 3 regions, agressive disease and poor prognosis
        # number of abnormalities: approximates complexity of cytogenetics (worse prognosis)
        # mosaic: multiple cell population, indicates clonal heterogeneity, helps understand tumor evolution and treatment resistance

        cyto = (
            cyto.str.lower()
                .str.replace(r'\[.*?\]', '', regex=True)
                .str.replace('onfish', '', regex=False)
                .str.strip()
        )

        clinical_df['cyto_normal'] = cyto.str.fullmatch(r'46,(xx|xy)').astype("Int64").where(cyto.notna(), pd.NA)
        clinical_df['cyto_complex'] = cyto.str.contains('complex', na=False).astype("Int64").where(cyto.notna(), pd.NA)
        clinical_df['monosomy_7'] = cyto.str.contains(r'-7', na=False).astype("Int64").where(cyto.notna(), pd.NA)
        clinical_df['trisomy_8'] = cyto.str.contains(r'\+8', na=False).astype("Int64").where(cyto.notna(), pd.NA)
        clinical_df['del_5q'] = cyto.str.contains(r'del\(5', na=False).astype("Int64").where(cyto.notna(), pd.NA)
        clinical_df['t_3_3'] = cyto.str.contains(r't\(3;3\)', na=False).astype("Int64").where(cyto.notna(), pd.NA)
        clinical_df['n_abnormalities'] = cyto.str.count(r'del|add|dic|der|inv|t\(|\+|-').where(cyto.notna(), pd.NA)
        clinical_df['cyto_mosaic'] = cyto.str.contains('/', na=False).astype("Int64").where(cyto.notna(), pd.NA)

        clinical_df = clinical_df.drop('CYTOGENETICS', axis=1)
        return clinical_df

    def _decode_chromosomes(self,molecular_df:pd.DataFrame) -> pd.DataFrame:

        # function that, for each patient, aggregates the number of genetic mutation he has
        # and for each chromosome pair, a 0 if it's not mutated and the sum of the VAFs associated if it is
        # like a one hot encoder but with VAFs information instead of a pure binary information (0/1)

        mol_agg_gen = (
            molecular_df
            .groupby("ID")
            .agg(
                nb_mutations=("GENE", "count"))
        )
        
        ID = molecular_df.index.tolist()
        CH=list(molecular_df["CHR"])
        VAF=list(molecular_df["VAF"])

        df_chrom=pd.DataFrame({
            "ID":ID,
            "Chromosomes":CH,
            "VAF":VAF,
        })

        result=df_chrom.pivot_table(index="ID", columns="Chromosomes",values="VAF", aggfunc="sum",fill_value=0).reset_index()

        molecular_df=result.merge(mol_agg_gen,on="ID",how="inner")
        molecular_df = molecular_df.set_index("ID")

        return molecular_df
    
    def _decode_genes(self,molecular_df:pd.DataFrame) -> pd.DataFrame:

        # function that, for each patient, aggregates the number of genetic mutation he has
        # and for each potential gene amongst the list of all mutated genes inside the training sample, a 0 if it's not mutated and the sum of the VAFs associated if it is
        # like a one hot encoder but with VAFs information instead of a pure binary information (0/1)
        # it also adds information about the impact of the mutations: if it contains "stop gained", "frameshift" or "splice", they are considered as high impact mutations
        # "stop gained": there's a premature stop, the gene often loses its function in its entirety (often affects genes suppressing tumors, allowing proliferation of cancerous cells)
        # "frameshift": non functional protein (often impacts genes controlling DNA repair)
        # "splice": affects RNA, changing how cells are processed.
        # here, we count the number of high impact mutations a patient has

        mol_agg_gen = (
            molecular_df
            .groupby("ID")
            .agg(
                nb_mutations=("GENE", "count"),
                max_vaf=("VAF", "max"))
        )
        
        high_impact_count = molecular_df[molecular_df["EFFECT"].isin(["stop_gained","frameshift","splice"])].groupby("ID")["GENE"].count()
        mol_agg_gen["high_impact_mutations"] = high_impact_count
        mol_agg_gen["high_impact_mutations"] = mol_agg_gen["high_impact_mutations"].fillna(0)


        ID = molecular_df.index.tolist()
        GEN=list(molecular_df["GENE"])
        VAF=list(molecular_df["VAF"])

        df_chrom=pd.DataFrame({
            "ID":ID,
            "Genes":GEN,
            "VAF":VAF,
        })

        result=df_chrom.pivot_table(index="ID", columns="Genes",values="VAF", aggfunc="sum",fill_value=0).reset_index()

        molecular_df=result.merge(mol_agg_gen,on="ID",how="inner")
        molecular_df = molecular_df.set_index("ID")

        return molecular_df

    def _aggregator(self, clinical_df: pd.DataFrame, molecular_df: pd.DataFrame) -> pd.DataFrame:
        if molecular_df is None:
            return clinical_df.copy()
        
        # Ensure join is explicit on index
        if clinical_df.index.dtype == molecular_df.index.dtype or 'ID' in clinical_df.columns:
            # try to join on index; if IDs are a column, prefer merge
            try:
                joined = clinical_df.join(molecular_df)
            except Exception:
                joined = clinical_df.reset_index().merge(molecular_df.reset_index(), left_on='ID', right_on='ID', how='left').set_index(clinical_df.index.name)
        else:
            joined = clinical_df.join(molecular_df)

        return joined

    def _categorize(self, df: pd.DataFrame):
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        binary_cols = ['cyto_normal', 'cyto_complex', 'monosomy_7', 'trisomy_8', 'del_5q', 't_3_3', 'cyto_mosaic']
        float_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return categorical_cols, binary_cols, float_cols

    def _drop_nan_target(self, df: pd.DataFrame, y: pd.DataFrame, molecular_df: pd.DataFrame):
        if y is None:
            return df, y, molecular_df

        valid_idx = y.dropna().index
        y_filtered = y.loc[valid_idx]

        df_filtered = df.loc[valid_idx] if df is not None else None
        clinical_filtered = None
        molecular_filtered = None

        if molecular_df is not None:
            # molecular_df often contains rows per mutation; filter by ID column if present
            if 'ID' in molecular_df.columns:
                molecular_filtered = molecular_df[molecular_df['ID'].isin(valid_idx)].copy()
            else:
                molecular_filtered = molecular_df.loc[molecular_df.index.intersection(valid_idx)]

        return df_filtered, y_filtered, molecular_filtered
    




# %%%%%%%%%%%%% === MAIN === %%%%%%%%%%%%%
if __name__ == "__main__":
    df = pd.read_csv("./data/raw/X_train/clinical_train.csv", index_col=0)
    maf_df = pd.read_csv("./data/raw/X_train/molecular_train.csv", index_col=0)
    target_df = pd.read_csv("./data/raw/target_train.csv", index_col=0)
    dtm = DataViewer(df, target_df)
    print(dtm.stats_analysis())
