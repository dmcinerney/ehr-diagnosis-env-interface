import streamlit as st
import pandas as pd
import numpy as np
from utils import *


st.set_page_config(layout="wide")
st.title('EHR Diagnosis Environment Visualizer')
st.button('Reset session', on_click=reset_session_state)
args = get_args('config.yaml')
dataset = args['data']['dataset']
st.write(f'Dataset: \"{dataset}\"')
splits = get_splits(args)
with st.sidebar:
    split = st.selectbox('Dataset Split', splits, index=splits.index('val') if 'val' in splits else 0)
    string_match_filter = st.text_area('String Match Filter (commas are \'and\'s, linebreaks are \'or\'s)')
    df = get_filtered_dataset(args, split, string_match_filter)
    if len(df) == 0:
        st.warning('No results after filtering.')
    instance_name = st.selectbox(f'Instances ({len(df)})', list(df.instance_name), on_change=reset_session_state)
    st.divider()
    st.write('Configuration')
    st.write(args)
    st.divider()
    spinner_container = st.container()
if len(df) == 0:
    st.stop()
container = st.empty()


env = get_environment()
# only does this when changing dataset with args, split, or filter
set_environment_instances(env, df, args, split, string_match_filter)
# st.write(str(env.model.model.lm_head.weight.device))
if 'reset' not in st.session_state.keys():
    # store the return value of reset for the current instance
    with st.spinner('Extracting information from reports to set up environment...'):
        st.session_state['reset'] = env.reset(
            options={
                'instance_index': np.where(df.instance_name == instance_name)[0].item()})
    # store the actions for the current instance
    st.session_state['actions'] = {}
    # store the return value of the step function for the current instance
    st.session_state['steps'] = {}
    st.session_state['cumulative_reward'] = 0


observation, info = st.session_state['reset']
reward = 0
for i in range(1000):
    with container.container():
        if i not in st.session_state['steps'].keys():
            st.subheader("Information")
            st.write(f'**timestep**: {i + 1}')
            st.write('**reward for previous action**: {}'.format(reward))
            st.write('**cumulative reward**: {}'.format(st.session_state['cumulative_reward']))
            for k, v in info.items():
                st.write('**{}**: {}'.format(k, v))
            if len(info['current_targets']) == 0:
                st.warning('No targets so this is a dead environment! You can move to another instance.')
            st.subheader("Evidence")
            if observation['evidence'].strip() != '':
                st.write(pd.read_csv(io.StringIO(observation['evidence'])))
            potential_diagnoses = pd.read_csv(io.StringIO(observation['potential_diagnoses'])).diagnoses.to_list()
            st.subheader("Action")
            st.write('Take an action by ranking potential diagnoses')
            if i not in st.session_state['actions'].keys():
                policy = st.selectbox(
                    'Choose how to pick an action', ['Choose your own action', 'Random policy'], key=f'select policy {i}')
                action = env.action_space.sample()
                if policy == 'Choose your own action':
                    action = [0] * len(potential_diagnoses)
                elif policy == 'Random policy':
                    with st.spinner('Sampling action'):
                        action = env.action_space.sample()
                action_df = pd.DataFrame({'rating': {d: a for d, a in zip(potential_diagnoses, action)}})
                if policy == 'Choose your own action':
                    edited_df = st.experimental_data_editor(action_df, key=f'action editor {i}')
                    action_dict = edited_df.to_dict()['rating']
                    action = [action_dict[d] for d in potential_diagnoses]
                else:
                    st.write(action_df)
            action_submitted = st.button('Submit Action', key=f'submit {i}')
            submit_button_container = st.container()
            st.subheader("Reports")
            filter_reports = st.checkbox('Filter Reports', key=f'filter reports {i}')
            run_extraction = st.checkbox('Run extraction', key=f'run extraction {i}')
            show_raw = st.checkbox('Show Extracted Raw Output', key=f'show raw {i}')
            display_report(
                filter_dataframe(
                    process_reports(observation['reports']), string_match_filter if filter_reports else ''),
                f'timestep {i + 1}')
            if not action_submitted:
                st.stop()
            with submit_button_container:
                with st.spinner('Taking step'):
                    st.session_state['actions'][i] = action
                    st.session_state['steps'][i] = env.step(action)
        observation, reward, terminated, truncated, info = st.session_state['steps'][i]
        st.session_state['cumulative_reward'] += reward
        if terminated or truncated:
            break
env.close()
st.write('Done! Environment was {}.'.format('terminated' if terminated else 'truncated'))
