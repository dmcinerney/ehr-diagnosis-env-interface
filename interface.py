from email.policy import default
from utils import *
import streamlit as st
import pandas as pd
import numpy as np
from ehr_diagnosis_agent.models.actor import InterpretableDirichletActor, InterpretableNormalActor, InterpretableBetaActor
import torch
from torch.distributions import Beta
from omegaconf import OmegaConf


actor_checkpoints = {
    # 'supervised_dirichlet': (
    #     '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230604_010745-6xacw8w1/files/ckpt_epoch=8_updates=986.pt',
    #     '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230604_010745-6xacw8w1/files/config.yaml'
    # ),
    # 'supervised_beta': (
    #     '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230614_095808-ov4baor2/files/ckpt_epoch=3_updates=34.pt',
    #     '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230614_095808-ov4baor2/files/config.yaml'
    # ),
    # 'supervised_beta_with_bias': (
    #     '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230614_134807-aaf16kc6/files/ckpt_epoch=23_updates=289.pt',
    #     '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230614_134807-aaf16kc6/files/config.yaml'
    # ),
    # 'supervised_beta_with_attn': (
    #     '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230614_183930-pj9ipm5h/files/ckpt_epoch=9_updates=92.pt',
    #     '/work/frink/mcinerney.de/ehr-diagnosis-agent/ehr_diagnosis_agent/wandb/run-20230614_183930-pj9ipm5h/files/config.yaml'
    # ),
    # 'supervised_beta_with_attn2': (
    #     '/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20230616_004559-fn9b7mmz/files/ckpt_epoch=5_updates=218.pt',
    #     '/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20230616_004559-fn9b7mmz/files/config.yaml'
    # ),
    # 'supervised_beta_with_attn3': (
    #     '/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20230616_122933-pp2dkyve/files/ckpt_epoch=16_updates=701.pt',
    #     '/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20230616_122933-pp2dkyve/files/config.yaml'
    # ),
    # 'supervised_beta_with_attn4': (
    #     '/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20230617_213722-5zuox2g6/files/ckpt_epoch=41_updates=1853.pt',
    #     '/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20230617_213722-5zuox2g6/files/config.yaml'
    # ),

    'mixed_beta_with_attn': (
        '/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20230712_215840-ktr1ito1/files/ckpt_epoch=16_updates=539.pt',
        '/scratch/mcinerney.de/ehr-diagnosis-agent-output/wandb/run-20230712_215840-ktr1ito1/files/config.yaml'
    ),
}
actory_types = {
    'normal': InterpretableNormalActor,
    'dirichlet': InterpretableDirichletActor,
    'beta': InterpretableBetaActor,
}


@st.cache_resource
def get_actor(actor_checkpoint):
    args = OmegaConf.load(actor_checkpoints[actor_checkpoint][1])
    actor = actory_types[args.actor.value.type](
        args.actor.value['{}_params'.format(args.actor.value.type)])
    state_dict = torch.load(actor_checkpoints[actor_checkpoint][0])['actor']
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


@st.cache_data
def sample_action(observation, actor_checkpoint, sample_type='mean'):
    actor = get_actor(actor_checkpoint)
    with torch.no_grad():
        parameter_votes_info = actor.get_dist_parameter_votes(observation)
        parameters_info = actor.votes_to_parameters(
            parameter_votes_info['diagnosis_embeddings'],
            parameter_votes_info['context_embeddings'],
            parameter_votes_info['param_votes'])
    dist = actor.parameters_to_dist(*parameters_info['params'])
    # action = dist.sample()
    if sample_type == 'mean':
        if isinstance(actor, InterpretableDirichletActor) or isinstance(actor, InterpretableBetaActor):
            # action = torch.log(dist.base_dist.mode)
            action = torch.log(dist.base_dist.mean)
            action_stddev = dist.base_dist.stddev
        else:
            # action = dist.mode
            action = dist.mean
            action_stddev = dist.stddev
    else:
        raise Exception
    return parameter_votes_info, parameters_info, action.detach().numpy(), action_stddev


st.set_page_config(layout="wide")
st.title('EHR Diagnosis Environment Visualizer')
st.button('Reset session', on_click=reset_episode_state)
jump_timestep_container = st.container()
args = get_args('config.yaml')
dataset = args['data']['dataset']
st.write(f'Dataset: \"{dataset}\"')
splits = get_splits(args)
with st.sidebar:
    def set_reset_instances():
        st.session_state['reset_instances'] = True
    if 'reset_instances' not in st.session_state.keys():
        set_reset_instances()
    split = st.selectbox(
        'Dataset Split', splits,
        index=splits.index('val') if 'val' in splits else 0,
        on_change=set_reset_instances)
    df = get_dataset(args, split)
    env = get_environment(args)
    if st.session_state['reset_instances']:
        with st.spinner('resetting instances and reloading cache'):
            set_environment_instances(env, df, args, split)
        st.session_state['reset_instances'] = False
    section_size = 30
    sl = st.selectbox(
        'Dataset Section',
        [f'{i+1}-{i+section_size}'
         if i + section_size < len(df) else f'{i+1}-{len(df)}'
         for i in range(0, len(df), section_size)])
    valid_instances_filter = get_valid_instances_filter(env, args, split, sl)
    num_valid = valid_instances_filter.sum()
    if num_valid == 0:
        st.warning('No results after filtering.')
    instance_name = st.selectbox(f'Instances ({num_valid})', list(
        df[valid_instances_filter].instance_name), on_change=reset_episode_state)
    options = ['Choose your own action']
    if len(actor_checkpoints) > 0:
        options.append('Trained Actor')
    policy = st.selectbox(
        'Choose how to pick an action', options)
    actor_checkpoint = st.selectbox(
        'Choose your actor', actor_checkpoints.keys(),
        disabled=not policy == 'Trained Actor')
    sort_by_attn = st.checkbox('Sort by attention', value=True)
    st.divider()
    st.write('Configuration')
    st.write(args)
    st.divider()
    spinner_container = st.container()
if num_valid == 0:
    st.warning('No example to show')
    st.stop()
container = st.empty()

# st.write(str(env.model.model.lm_head.weight.device))
instance_index = np.where(df.instance_name == instance_name)[0].item()
if 'episode' not in st.session_state.keys():
    st.session_state['episode'] = {}
    # store the return value of reset for the current instance
    with st.spinner('Extracting information from reports to set up environment...'):
        st.session_state['episode']['reset'] = env.reset(
            options={'instance_index': instance_index})
    # store the actions for the current instance
    st.session_state['episode']['actions'] = {}
    # store the return value of the step function for the current instance
    st.session_state['episode']['steps'] = {}
    st.session_state['episode']['skip_to'] = None
observation, info = st.session_state['episode']['reset']
with jump_timestep_container:
    value = len(st.session_state['episode']['steps']) + \
        1 if st.session_state['episode']['skip_to'] is None else st.session_state['episode']['skip_to']
    jump_to = st.number_input(
        'Jump to a step', min_value=1, max_value=info['max_timesteps'], value=value)
    st.button('Jump', on_click=JumpTo(env, jump_to, instance_index))
reward = 0
i = 0
terminated, truncated = False, False


def display_state(observation, info, reward):
    st.subheader("Information")
    st.write(f'**timestep**: {i + 1}')
    st.write('**report**: {}'.format(info['current_report'] + 1))
    st.write(
        f'**is_truncated**: {env.is_truncated(observation, info)}')
    st.write(
        '**evidence_is_retrieved**: {}'.format(observation['evidence_is_retrieved']))
    with st.expander('Environment secrets'):
        st.write('**reward for previous action**: {}'.format(reward))
        cumulative_reward = sum(
            [reward for _, reward, _, _, _ in st.session_state['episode']['steps'].values()])
        st.write('**cumulative reward**: {}'.format(cumulative_reward))
        for k, v in info.items():
            if k == 'current_report':
                continue
            st.write('**{}**: {}'.format(k, v))
    if len(info['current_targets']) == 0:
        st.warning(
            'No targets so this is a dead environment! You can move to another instance.')


def display_beta_actor_action(
        observation, parameter_votes_info, parameters_info,
        checkpoint_args, index, compare_to, index2):
    # st.write('**concentration1**: {:.1f}'.format(parameters[0][index]))
    # st.write('**concentration0**: {:.1f}'.format(parameters[1][index]))
    concentration1_votes = parameter_votes_info['param_votes'][:, :, 0]
    concentration0_votes = parameter_votes_info['param_votes'][:, :, 1]
    context_strings = parameter_votes_info['context_strings']
                        # TODO: add attention info if it exists
    concentrations1 = torch.nn.functional.softplus(
        concentration1_votes) + \
        checkpoint_args.actor.value.beta_params.concentration_min
    concentrations0 = torch.nn.functional.softplus(
        concentration0_votes) + \
        checkpoint_args.actor.value.beta_params.concentration_min
    dist_means = [Beta(a, b).mean for a, b in zip(
        concentrations1[index], concentrations0[index])]
    dist_stddevs = [Beta(a, b).stddev for a, b in zip(
        concentrations1[index], concentrations0[index])]
    dist_means2 = [Beta(a, b).mean for a, b in zip(
        concentrations1[index2], concentrations0[index2])] \
        if compare_to else [None] * len(dist_means)
    evidence_concentrations = list(zip(
        context_strings, range(len(context_strings)), concentrations1[index],
        concentrations0[index], dist_means, dist_stddevs, dist_means2))
    def key(x): return -x[4]
    if 'context_attn_weights' in parameters_info.keys() and sort_by_attn:
        def key(x):
            return -parameters_info['context_attn_weights'][index, x[1]]
    evidence_concentrations = sorted(evidence_concentrations, key=key)
    for ec in evidence_concentrations:
        evidence = ("\"" + ec[0] + "\"") \
            if ec[1] > 0 or observation['evidence_is_retrieved'] else \
            'the report'
        if compare_to:
            attn_string = "" \
                if 'context_attn_weights' not in parameters_info.keys() else \
                ", attn={:.0f}%,{:.0f}%".format(
                    parameters_info['context_attn_weights'][
                        index, ec[1]].item() * 100,
                    parameters_info['context_attn_weights'][
                        index2, ec[1]].item() * 100,)
            st.write('- (vote={}{:.1f}%{}) {}'.format(
                '+' if ec[4] > ec[6] else '', (ec[4] - ec[6]) * 100,
                attn_string, evidence))
        else:
            attn_string = "" \
                if 'context_attn_weights' not in parameters_info.keys() else \
                ", attn={:.0f}%".format(
                parameters_info['context_attn_weights'][
                    index, ec[1]].item() * 100)
            st.write('- (vote={:.1f}%±{:.1f}{}) {}'.format(
                ec[4] * 100, ec[5] * 100, attn_string, evidence))


def display_diritchlet_actor_action(
        observation, parameter_votes_info, parameters_info, checkpoint_args,
        index, compare_to, index2, diagnosis):
    concentration = parameters_info['params'][0]
    st.write('**average of all concentrations**: {:.1f}'.format(
        concentration.mean()))
    st.write('**{} concentration**: {:.1f}'.format(
        diagnosis, concentration[index]))
    concentration_votes = parameter_votes_info['param_votes'][:, :, 0]
    context_strings = parameter_votes_info['context_strings']
    # TODO: add attention info if it exists
    concentrations = torch.nn.functional.softplus(concentration_votes) + \
        checkpoint_args.actor.value.dirichlet_params.concentration_min
    evidence_concentrations = list(zip(
        context_strings, range(len(context_strings)), concentrations[index],
        concentrations[index2]))
    def key(x): return -x[2]
    if 'context_attn_weights' in parameters_info.keys() and sort_by_attn:
        def key(x):
            return -parameters_info['context_attn_weights'][index, x[1]]
    evidence_concentrations = sorted(evidence_concentrations, key=key)
    for ec in evidence_concentrations:
        evidence = ("\"" + ec[0] + "\"") \
            if ec[1] > 0 or observation['evidence_is_retrieved'] else \
            'the report'
        if compare_to:
            attn_string = "" \
                if 'context_attn_weights' not in parameters_info.keys() else \
                ", attn={:.0f}%,{:.0f}%".format(
                    parameters_info[
                        'context_attn_weights'][index, ec[1]].item() * 100,
                    parameters_info[
                        'context_attn_weights'][index2, ec[1]].item() * 100)
            st.write('- (vote={}{:.1f}{}) {}'.format(
                '+' if ec[2] > ec[3] else '', ec[2] - ec[3], attn_string,
                evidence))
        else:
            attn_string = "" \
                if 'context_attn_weights' not in parameters_info.keys() else \
                ", attn={:.0f}%".format(
                    parameters_info[
                        'context_attn_weights'][index, ec[1]].item() * 100)
            st.write('- (vote={:.1f}{}) {}'.format(
                ec[2], attn_string, evidence))
    return key


def display_action_evidence(
        observation, parameter_votes_info, parameters_info, checkpoint_args,
        diagnosis, rating, stddev, index, compare_to, diagnosis2, rating2,
        stddev2, index2):
    if compare_to:
        st.write('### {} - {} (delta={}{:.1f}%)'.format(
            diagnosis, diagnosis2, '+' if rating < rating2 else '',
            (rating - rating2) * 100))
    else:
        st.write('### {} (rating={:.1f}%±{:.1f})'.format(
            diagnosis, rating * 100, stddev * 100))
    if isinstance(get_actor(actor_checkpoint), InterpretableDirichletActor):
        display_diritchlet_actor_action(
            observation, parameter_votes_info, parameters_info,
            checkpoint_args, index, compare_to, index2, diagnosis)
    elif isinstance(get_actor(actor_checkpoint), InterpretableBetaActor):
        return display_beta_actor_action(
            observation, parameter_votes_info, parameters_info,
            checkpoint_args, index, compare_to, index2)
    elif isinstance(get_actor(actor_checkpoint), InterpretableNormalActor):
        raise NotImplementedError


while not (terminated or truncated):
    with container.container():
        if i not in st.session_state['episode']['steps'].keys():
            display_state(observation, info, reward)
            st.subheader("Reports")
            display_report(
                process_reports(observation['reports']),
                f'timestep {i + 1}')
            st.subheader("Evidence")
            if observation['evidence'].strip() != '':
                st.write(pd.read_csv(io.StringIO(observation['evidence'])))
            else:
                st.write('No evidence yet!')
            options = pd.read_csv(io.StringIO(
                observation['options'])).apply(
                lambda r: f'{r.option} ({r.type})', axis=1).to_list()
            st.subheader("Action")
            st.write('Take an action by ranking potential diagnoses')
            if policy == 'Choose your own action':
                action = [0] * len(options)
                action = np.array(action, dtype=float)
                action_df = pd.DataFrame(
                    {'rating': {d: a for d, a in zip(options, action)}})
                edited_df = st.experimental_data_editor(
                    action_df, key=f'action editor {i}')
                # edited_df = st.data_editor(action_df, key=f'action editor {i}')
                action_dict = edited_df.to_dict()['rating']
                action = [action_dict[d] for d in options]
            elif policy == 'Trained Actor':
                parameter_votes_info, parameters_info, action, action_stddev = sample_action(
                    observation, actor_checkpoint)
                action = np.array(action, dtype=float)
                checkpoint_args = OmegaConf.load(
                    actor_checkpoints[actor_checkpoint][1])
                if checkpoint_args.actor.value.type == 'beta':
                    ratings = torch.exp(torch.tensor(action))
                    st.write(
                        'This actor\'s ratings are independent. (Probabilities do not sum to 1.)')
                else:
                    ratings = torch.softmax(torch.tensor(action), 0).numpy()
                    st.write(
                        'This actor\'s ratings are not independent. (Probabilities sum to 1.)')
                potential_diagnoses_ratings_indices = list(
                    zip(options, ratings, action_stddev, range(len(action))))
                potential_diagnoses_ratings_indices = sorted(
                    potential_diagnoses_ratings_indices, key=lambda x: -x[1])
                c1, c2 = st.columns(2)

                def format_func(x):
                    d, r, stddev, _ = potential_diagnoses_ratings_indices[x]
                    return '({:.1f}%±{:.1f}) {}'.format(r * 100, stddev * 100, d)
                with c1:
                    diagnosis_index = st.radio(
                        'Choose an option to see what impacted the score.',
                        range(len(potential_diagnoses_ratings_indices)),
                        format_func=format_func,
                        key=f'diagnosis to view {i}')
                    compare_to = st.checkbox(
                        'Compare option with...', key=f'compare to {i}')
                    diagnosis, rating, stddev, index = \
                        potential_diagnoses_ratings_indices[
                            diagnosis_index]
                    if compare_to:
                        diagnosis_index2 = st.radio(
                            'Choose another option to compare to.',
                            range(len(potential_diagnoses_ratings_indices)),
                            format_func=format_func,
                            key=f'diagnosis to compare to {i}')
                        diagnosis2, rating2, stddev2, index2 = \
                            potential_diagnoses_ratings_indices[
                                diagnosis_index2]
                    else:
                        diagnosis2, rating2, stddev2, index2 = \
                            None, None, None, None
                with c2:
                    display_action_evidence(
                        observation, parameter_votes_info, parameters_info,
                        checkpoint_args, diagnosis, rating, stddev, index,
                        compare_to, diagnosis2, rating2, stddev2, index2)
            else:
                raise Exception
            action_submitted = st.button('Submit Action', key=f'submit {i}')
            skip_past = st.session_state['episode']['skip_to'] is not None and \
                len(st.session_state['episode']['steps']) + \
                1 < st.session_state['episode']['skip_to']
            if not action_submitted and not skip_past:
                st.session_state['episode']['skip_to'] = None
                st.stop()
            with st.spinner('Taking step'):
                st.session_state['episode']['actions'][i] = action
                st.session_state['episode']['steps'][i] = env.step(action)
        observation, reward, terminated, truncated, info = st.session_state['episode']['steps'][i]
        i += 1
with container.container():
    display_state(observation, info, reward)
    st.write('Done! Environment was {}.'.format(
        'terminated' if terminated else 'truncated'))
