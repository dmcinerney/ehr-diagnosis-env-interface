from omegaconf import OmegaConf
import pandas as pd
import os
import streamlit as st
import io
import ehr_diagnosis_env
from ehr_diagnosis_env.utils import get_model_interface
from sentence_transformers import SentenceTransformer
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


@st.cache_data
def get_valid_instances_filter(_env, args, split, sl, **kwargs):
    cached_instances = _env.get_cached_instances()
    valid_instances = set()
    minimum, maximum = [int(x) for x in sl.split('-')]
    cached_instances = [i for i in cached_instances if i + 1 >= minimum and i + 1 <= maximum]
    for i in stqdm(cached_instances, total=len(cached_instances)):
        obs, info = _env.reset(options=get_reset_options(args, i))
        terminated = _env.is_terminated(obs, info)
        truncated = _env.is_truncated(obs, info)
        if not (terminated or truncated):
            valid_instances.add(i)
    filter = pd.Series(range(_env.num_examples())).apply(
        lambda x: x in valid_instances)
    return filter


def reset_episode_state():
    del st.session_state['episode']


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
    report_name = st.selectbox('Choose a report', report_names, key=key, index=len(report_names) - 1)
    report_row = reports_df[reports_df.report_name == report_name].iloc[0]
    st.write(f'Description: {report_row.description}')
    st.divider()
    st.write(report_row.text)


@st.cache_resource
def get_env_models(args):
    return get_model_interface('google/flan-t5-xxl'), \
        SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def get_environment(args, split, _instances, _llm_interface, _fmm_interface,
                    **kwargs):
    return gymnasium.make(
        'ehr_diagnosis_env/EHRDiagnosisEnv-v0',
        instances=_instances,
        cache_path=args['data'][f'{split}_cache_path'],
        llm_name_or_interface=_llm_interface,
        fmm_name_or_interface=_fmm_interface,
        progress_bar=stqdm,
        **kwargs
    )


def get_reset_options(args, i):
    options = {'instance_index': i}
    if args['data']['max_reports_considered'] is not None:
        options['max_reports_considered'] = \
            args['data']['max_reports_considered']
    return options


class JumpTo:
    def __init__(self, args, env, jump_to, instance_index):
        self.args = args
        self.env = env
        self.jump_to = jump_to - 1
        self.instance_index = instance_index

    def __call__(self):
        current_timestep = len(st.session_state['episode']['steps'])
        if self.jump_to >= current_timestep:
            st.session_state['episode']['skip_to'] = self.jump_to + 1
            return
        st.session_state['episode']['skip_to'] = None
        for i in range(current_timestep - 1, self.jump_to - 1, -1):
            _, reward, _, _, _ = st.session_state['episode']['steps'][i]
            del st.session_state['episode']['steps'][i]
            del st.session_state['episode']['actions'][i]
        with st.spinner('re-running environment to rewind'):
            # we can do this because we know the environment is not stochastic
            self.env.reset(options=get_reset_options(self.args, self.instance_index))
            for step in range(self.jump_to):
                self.env.step(st.session_state['episode']['actions'][step])
