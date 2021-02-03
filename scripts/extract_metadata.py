from pathlib import Path
from typing import Dict, Tuple, List, Union, Optional

import click
from click_pathlib import Path as cPath
import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

cols_imdb = ['age', 'gender', 'path', 'names', 'face_score1', 'face_score2']
cols_wiki = ['age', 'gender', 'path', 'face_score1', 'face_score2']

imdb_mat = 'data/imdb.mat'
wiki_mat = 'data/wiki.mat'


def _load_and_preprocess(path: Path, name: str) -> np.ndarray:
    mat = loadmat(str(path))
    return mat[name]


def _extract_info(raw_meta: np.ndarray, include_name: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    photo_taken = raw_meta[0][0][1][0]
    full_path = raw_meta[0][0][2][0]
    gender = raw_meta[0][0][3][0]
    if include_name:
        name = np.asarray([name[0] for name in raw_meta[0][0][4][0]])
    else:
        name = None
    face_score1 = raw_meta[0][0][6][0]
    face_score2 = raw_meta[0][0][7][0]
    return photo_taken, full_path, gender, name, face_score1, face_score2


def _expand_path(filename: Union[List[str], np.ndarray], parent: Path) -> str:
    return str(parent / filename[0])


def _create_paths(full_paths: np.ndarray, parent: Path) -> List[str]:
    paths = []
    for path in tqdm(full_paths, desc=f"creating filepath for {str(parent)}"):
        paths.append(_expand_path(path, parent))
    return paths


def _create_genders(gender_raw: np.ndarray) -> List[str]:
    genders = []
    for n in tqdm(range(len(gender_raw)), "parsing genders"):
        if gender_raw[n] == 1:
            genders.append('male')
        else:
            genders.append('female')
    return genders


def _get_imdb_dob2(paths: List[str]) -> List[str]:
    dob_raw: pd.Series = pd.Series([file.split("_")[3] for file in paths])
    dob = pd.to_datetime(dob_raw)
    return dob


def _get_imdb_dob(paths: List[str]) -> List[str]:
    imdb_dob = []

    for file in tqdm(paths, "parsing imdb dob"):
        temp = file.split('_')[3]
        temp = temp.split('-')
        if len(temp[1]) == 1:
            temp[1] = '0' + temp[1]
        if len(temp[2]) == 1:
            temp[2] = '0' + temp[2]

        if temp[1] == '00':
            temp[1] = '01'
        if temp[2] == '00':
            temp[2] = '01'

        imdb_dob.append('-'.join(temp))
    return imdb_dob


def _get_wiki_dob(wiki_path: List[str]) -> List[str]:
    wiki_dob = []
    for file in tqdm(wiki_path, "parsing dob for wiki"):
        wiki_dob.append(file.split('_')[2])
    return wiki_dob


def _get_age(dob: List[str], photo_taken: np.ndarray) -> List[int]:
    age = []
    date_of_birst = []
    errors = 0
    for i in tqdm(range(len(dob)), "parsing age"):
        try:
            dob_date_str = dob[i][0:10].replace("-00", "-01")
            d1 = date.datetime.strptime(dob_date_str, '%Y-%m-%d')
            d2 = date.datetime.strptime(str(photo_taken[i]), '%Y')
            rdelta = relativedelta(d2, d1)
            diff = rdelta.years

        except Exception as ex:
            #print(ex)
            errors += 1
            diff = -1
        age.append(diff)
    print(f"{errors} out of {len(dob)} files erronious ({round(100*(errors/len(dob)), 2)}%)")
    return age


def _create_meta(final_imdb_df: pd.DataFrame, final_wiki_df: pd.DataFrame, out: Path) -> pd.DataFrame:
    print("Removing faceless images")
    meta = pd.concat((final_imdb_df, final_wiki_df))
    meta = meta[meta['face_score1'] != '-inf']
    #meta = meta[meta['face_score2'] == 'nan']
    #meta = meta.drop(['face_score1', 'face_score2'], axis=1)
    #meta = meta.sample(frac=1)
    print("Before:", len(final_imdb_df)+len(final_wiki_df), "After:" ,len(meta), "Perc Loss:", 100-round(100*(len(meta)/(len(final_imdb_df)+len(final_wiki_df))), 2), "%")
    return meta


@click.command()
@click.option("--imdb_path", type=cPath(), required=False, default='../data/imdb.mat')
@click.option("--wiki_path", type=cPath(), required=False, default='../data/wiki.mat')
@click.option("--imdb_img_source", type=cPath(), required=False, default='imdb_crop/')
@click.option("--wiki_img_source", type=cPath(), required=False, default='wiki_crop/')
@click.option("--out", type=cPath(), required=False, default="meta.csv")
def main(imdb_path: Path, wiki_path: Path, imdb_img_source: Path, wiki_img_source: Path, out: Path):

    imdb = _load_and_preprocess(imdb_path, "imdb")
    wiki = _load_and_preprocess(wiki_path, "wiki")

    print("Modifying IMDB dataset")
    imdb_photo_taken, imdb_full_path, imdb_gender, names, imdb_face_score1, imdb_face_score2 = _extract_info(imdb)
    imdb_path = _create_paths(imdb_full_path, imdb_img_source)
    imdb_genders = _create_genders(imdb_gender)
    imdb_dob = _get_imdb_dob(imdb_path)
    imdb_age = _get_age(imdb_dob, imdb_photo_taken)
    final_imdb = np.vstack((imdb_age, imdb_genders, imdb_path, names, imdb_face_score1, imdb_face_score2)).T
    final_imdb_df = pd.DataFrame(final_imdb)
    final_imdb_df.columns = cols_imdb

    print("Modifying WIKI dataset")
    wiki_photo_taken, wiki_full_path, wiki_gender, _, wiki_face_score1, wiki_face_score2 = _extract_info(wiki, False)
    wiki_path = _create_paths(wiki_full_path, wiki_img_source)
    wiki_genders = _create_genders(wiki_gender)
    wiki_dob = _get_wiki_dob(wiki_path)
    wiki_age = _get_age(wiki_dob, wiki_photo_taken)
    final_wiki = np.vstack((wiki_age, wiki_genders, wiki_path, wiki_face_score1, wiki_face_score2)).T
    final_wiki_df = pd.DataFrame(final_wiki)
    final_wiki_df.columns = cols_wiki

    meta = _create_meta(final_imdb_df, final_wiki_df, out)
    print("IMDB Dataset Rows", len(final_imdb_df), "Wiki Dataset Rows:", len(final_wiki_df))
    print("Total", len(meta))
    print()
    meta.to_csv(out, index=False)


if __name__ == "__main__":
    main()
