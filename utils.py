from omegaconf import OmegaConf
import pandas as pd
import os
import streamlit as st
import io
import ehr_diagnosis_env
import gymnasium
from stqdm import stqdm


# Get args from a config file and override with cli
@st.cache_data
def get_args(config_file):
    args_from_cli = OmegaConf.from_cli()
    args_from_yaml = OmegaConf.load(config_file)
    return OmegaConf.to_container(OmegaConf.merge(args_from_yaml, args_from_cli))


@st.cache_data
def get_splits(args):
    files = os.listdir(os.path.join(args['data']['path'], args['data']['dataset']))
    return [split for split in ['train', 'val', 'test'] if f'{split}.data' in files]


def get_instance_name(args, row):
    instance = row.to_dict()
    reports = pd.read_csv(io.StringIO(instance['reports']))
    return f'Instance {row.name + 1} (patient {reports.iloc[0].patient_id}, ' \
           f'{len(reports)} reports)'


@st.cache_data
def get_dataset(args, split):
    df = pd.read_csv(os.path.join(args['data']['path'], args['data']['dataset'], f'{split}.data'), compression='gzip')
    df['instance_name'] = df.apply(lambda r: get_instance_name(args, r), axis=1)
    return df


def filter_dataframe(df, string_match_filter):
    if string_match_filter == '':
        return df
    def filter(r):
        term_groups = [
            [term.strip().lower() in str(r.to_dict()).lower() for term in group.split(',')]
            for group in string_match_filter.split('\n')]
        return any([all(group) for group in term_groups])
    return df[df.apply(filter, axis=1)]


@st.cache_data
def get_filtered_dataset(args, split, string_match_filter):
    df = get_dataset(args, split)
    df = filter_dataframe(df, string_match_filter)
    return df


def reset_session_state():
    for k in st.session_state.keys():
        del st.session_state[k]


def get_report_name(row):
    return '{}. {} - {} ({})'.format(row.name + 1, row.hadm_id, row.report_type, row.date.strftime('%m/%d/%Y'))


def process_reports(reports):
    reports = pd.read_csv(io.StringIO(reports), parse_dates=['date'])
    reports['report_name'] = reports.apply(get_report_name, axis=1)
    return reports


def display_report(reports_df, key):
    if len(reports_df) == 0:
        st.write('No reports to display.')
        return None, None
    report_names = reports_df.report_name
    report_name = st.selectbox('Choose a report', report_names, key=key)
    report_row = reports_df[reports_df.report_name == report_name].iloc[0]
    st.write(f'Description: {report_row.description}')
    st.divider()
    st.write(report_row.text)


@st.cache_resource
def get_environment():
    return gymnasium.make(
        'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
        model_name='google/flan-t5-xl',
        # model_name='google/flan-t5-xxl'
        progress_bar=stqdm,
    )

@st.cache_resource
def set_environment_instances(_env, _df, args, split, string_match_filter):
    _env.set_instances(_df)
