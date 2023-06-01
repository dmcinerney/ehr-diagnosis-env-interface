from utils import *
import streamlit as st
import pandas as pd
import numpy as np
from ehr_diagnosis_agent.models.actor import InterpretableDirichletActor, InterpretableNormalActor
import torch
from omegaconf import OmegaConf


actor_checkpoints = {
    'supervised': (
        '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230530_203831-q71s5dsd/files/ckpt_epoch=30_updates=1448.pt',
        '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230530_203831-q71s5dsd/files/config.yaml'
    ),
    'supervised2': (
        '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230530_203831-q71s5dsd/files/ckpt_epoch=85_updates=4020.pt',
        '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230530_203831-q71s5dsd/files/config.yaml'
    ),
    'supervised3': (
        '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230601_092842-fzso76y6/files/ckpt_epoch=1_updates=236.pt',
        '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230601_092842-fzso76y6/files/config.yaml'
    )
}
@st.cache_resource
def get_actor(actor_checkpoint):
    args = OmegaConf.load(actor_checkpoints[actor_checkpoint][1])
    if args.actor.value.type == 'normal':
        actor = InterpretableNormalActor(args.actor.value.normal_params)
    elif args.actor.value.type == 'dirichlet':
        actor = InterpretableDirichletActor(args.actor.value.dirichlet_params)
    else:
        raise Exception
    state_dict = torch.load(actor_checkpoints[actor_checkpoint][0])['actor']
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor
@st.cache_data
def sample_action(observation, actor_checkpoint):
    actor = get_actor(actor_checkpoint)
    with torch.no_grad():
        votes_and_context_strings = actor.get_dist_parameter_votes_and_evidence_strings(observation)
        parameters = actor.votes_to_parameters(*votes_and_context_strings[:-1])
        dist = actor.parameters_to_dist(*parameters)
    # action = dist.sample()
    if isinstance(actor, InterpretableDirichletActor):
        # st.write(dist.base_dist.mode)
        action = torch.log(dist.base_dist.mode)
    else:
        action = dist.mode
    return votes_and_context_strings, parameters, action.detach().numpy()


st.set_page_config(layout="wide")
st.title('EHR Diagnosis Environment Visualizer')
st.button('Reset session', on_click=reset_session_state)
reward_button = st.container()
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
    options = ['Choose your own action']
    if len(actor_checkpoints) > 0:
        options.append('Trained Actor')
    policy = st.selectbox(
        'Choose how to pick an action', options)
    if policy == 'Trained Actor':
        actor_checkpoint = st.selectbox('Choose your actor', actor_checkpoints.keys())
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
def rewind_one_step():
    i = len(st.session_state['steps']) - 1
    if i < 0:
        return
    _, reward, _, _, _ = st.session_state['steps'][i]
    st.session_state['cumulative_reward'] -= reward
    del st.session_state['steps'][i]
    del st.session_state['actions'][i]
    with st.spinner('re-running environment to rewind'):
        # we can do this because we know the environment is not stochastic
        env.reset(options={'instance_index': np.where(df.instance_name == instance_name)[0].item()})
        for step in range(i):
            env.step(st.session_state['actions'][step])
reward_button.button('Rewind one step', on_click=rewind_one_step)


observation, info = st.session_state['reset']
reward = 0
i = 0
terminated, truncated = False, False
while not (terminated or truncated):
    with container.container():
        if i not in st.session_state['steps'].keys():
            st.subheader("Information")
            st.write(f'**timestep**: {i + 1}')
            st.write('**report**: {}'.format(info['current_report'] + 1))
            st.write(f'**is_truncated**: {env.is_truncated(observation, info)}')
            st.write('**evidence_is_retrieved**: {}'.format(observation['evidence_is_retrieved']))
            with st.expander('Environment secrets'):
                st.write('**reward for previous action**: {}'.format(reward))
                st.write('**cumulative reward**: {}'.format(st.session_state['cumulative_reward']))
                for k, v in info.items():
                    if k == 'current_report':
                        continue
                    st.write('**{}**: {}'.format(k, v))
            if len(info['current_targets']) == 0:
                st.warning('No targets so this is a dead environment! You can move to another instance.')
            st.subheader("Reports")
            filter_reports = st.checkbox('Filter Reports', key=f'filter reports {i}')
            display_report(
                filter_dataframe(
                    process_reports(observation['reports']), string_match_filter if filter_reports else ''),
                f'timestep {i + 1}')
            st.subheader("Evidence")
            if observation['evidence'].strip() != '':
                st.write(pd.read_csv(io.StringIO(observation['evidence'])))
            else:
                st.write('No evidence yet!')
            potential_diagnoses = pd.read_csv(io.StringIO(observation['potential_diagnoses'])).diagnoses.to_list()
            st.subheader("Action")
            st.write('Take an action by ranking potential diagnoses')
            if policy == 'Choose your own action':
                action = [0] * len(potential_diagnoses)
            elif policy == 'Trained Actor':
                votes_and_context_strings, parameters, action = sample_action(observation, actor_checkpoint)
            else:
                raise Exception
            action = np.array(action, dtype=float)
            if policy == 'Choose your own action':
                action_df = pd.DataFrame({'rating': {d: a for d, a in zip(potential_diagnoses, action)}})
                edited_df = st.experimental_data_editor(action_df, key=f'action editor {i}')
                action_dict = edited_df.to_dict()['rating']
                action = [action_dict[d] for d in potential_diagnoses]
            elif policy == 'Trained Actor':
                potential_diagnoses_ratings_indices = list(zip(
                    # potential_diagnoses, action, range(len(action))))
                    potential_diagnoses, torch.softmax(torch.tensor(action), 0).numpy(), range(len(action))))
                potential_diagnoses_ratings_indices = sorted(
                    potential_diagnoses_ratings_indices, key=lambda x: -x[1])
                c1, c2 = st.columns(2)
                def format_func(x):
                    d, r, _ = potential_diagnoses_ratings_indices[x]
                    return '{}\t\t\t({})'.format(d, r)
                with c1:
                    diagnosis_index = st.radio(
                        'Choose a diagnosis to see what impacted the score.',
                        range(len(potential_diagnoses_ratings_indices)),
                        format_func=format_func,
                        key=f'diagnosis to view {i}')
                diagnosis, rating, index = potential_diagnoses_ratings_indices[diagnosis_index]
                with c2:
                    st.write(f'### {diagnosis} (rating={rating})')
                    if isinstance(get_actor(actor_checkpoint), InterpretableDirichletActor):
                        st.write('**average concentration**: {}'.format(parameters[0].mean()))
                        st.write('**{} concentration**: {}'.format(diagnosis, parameters[0][index]))
                        concentration_votes, context_strings = votes_and_context_strings
                        concentrations = torch.nn.functional.softplus(concentration_votes) + .2
                        evidence_concentrations = list(zip(
                            context_strings, concentrations[index], range(len(context_strings))))
                        evidence_concentrations = sorted(evidence_concentrations, key=lambda x: -x[1])
                        for ec in evidence_concentrations:
                            evidence = ("\"" + ec[0] + "\"") \
                                if ec[2] > 0 or observation['evidence_is_retrieved'] else 'the report'
                            st.write(f'- ({ec[1]}) {evidence}')
                    elif isinstance(get_actor(actor_checkpoint), InterpretableNormalActor):
                        means, log_stddevs, context_strings = votes_and_context_strings
            action_submitted = st.button('Submit Action', key=f'submit {i}')
            submit_button_container = st.container()
            if not action_submitted:
                st.stop()
            elif submit_button_container is not None:
                with submit_button_container:
                    with st.spinner('Taking step'):
                        st.session_state['actions'][i] = action
                        st.session_state['steps'][i] = env.step(action)
        observation, reward, terminated, truncated, info = st.session_state['steps'][i]
        st.session_state['cumulative_reward'] += reward
        i += 1
env.close()
with container:
    st.write('Done! Environment was {}.'.format('terminated' if terminated else 'truncated'))
