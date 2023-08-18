from email.policy import default
from utils import *
import streamlit as st
import pandas as pd
import numpy as np
from ehr_diagnosis_agent.models.actor import InterpretableDirichletActor, \
    InterpretableNormalActor, InterpretableBetaActor, InterpretableDeltaActor
from ehr_diagnosis_agent.models.observation_embedder import \
    BertObservationEmbedder
import torch
from torch.distributions import Beta, Categorical
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns


actor_types = {
    'normal': InterpretableNormalActor,
    'dirichlet': InterpretableDirichletActor,
    'beta': InterpretableBetaActor,
    'delta': InterpretableDeltaActor,
}
recommended_reward_types = {
    'normal': ['continuous_dependent', 'ranking'],
    'dirichlet': ['continuous_dependent', 'ranking'],
    'beta': ['continuous_independent', 'ranking'],
    'delta': ['continuous_independent', 'continuous_dependent', 'ranking'],
}


@st.cache_data
def get_checkpoint_args(args, actor_checkpoint):
    return OmegaConf.load(args['models'][actor_checkpoint][1])


@st.cache_resource
def get_actor(args, actor_checkpoint):
    checkpoint_args = get_checkpoint_args(args, actor_checkpoint)
    actor_params = checkpoint_args.actor.value['{}_params'.format(
        checkpoint_args.actor.value.type)]
    actor_params.update(checkpoint_args.actor.value['shared_params'])
    actor = actor_types[checkpoint_args.actor.value.type](actor_params)
    state_dict = torch.load(args['models'][actor_checkpoint][0])['actor']
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


@st.cache_data
def sample_action(args, observation, actor_checkpoint, sample_type='mean'):
    actor = get_actor(args, actor_checkpoint)
    with torch.no_grad():
        parameter_votes_info = actor.get_dist_parameter_votes(observation)
        parameters_info = actor.votes_to_parameters(
            parameter_votes_info['diagnosis_embeddings'],
            parameter_votes_info['context_embeddings'],
            parameter_votes_info['param_votes'])
    dist = actor.parameters_to_dist(*parameters_info['params'])
    # action = dist.sample()
    if sample_type == 'mean':
        if isinstance(actor, InterpretableDirichletActor) or isinstance(
                actor, InterpretableBetaActor):
            # action = torch.log(dist.base_dist.mode)
            action = torch.log(dist.base_dist.mean)
            action_stddev = dist.base_dist.stddev
        else:
            # action = dist.mode
            action = dist.mean
            action_stddev = dist.stddev
    else:
        raise Exception
    return parameter_votes_info, parameters_info, action.detach().numpy(), \
        action_stddev


st.set_page_config(layout="wide")
st.title('EHR Diagnosis Environment Visualizer')
st.button('Reset session', on_click=reset_episode_state)
jump_timestep_container = st.container()
args = get_args('config.yaml')
dataset = args['data']['dataset']
st.write(f'Dataset: \"{dataset}\"')
splits = get_splits(args)
with st.sidebar:
    annotate = st.checkbox('Annotate')
    options = ['Choose your own action']
    if len(args['models']) > 0:
        options.append('Trained Actor')
    policy = st.selectbox(
        'Choose how to pick an action', options)
    actor_checkpoint = st.selectbox(
        'Choose your actor', args['models'].keys(),
        disabled=not policy == 'Trained Actor')
    checkpoint_args = get_checkpoint_args(args, actor_checkpoint)
    st.write(f'Reward type: {checkpoint_args.env.value.reward_type}')
    split = st.selectbox(
        'Dataset Split', splits,
        index=splits.index('val') if 'val' in splits else 0,
        on_change=reset_episode_state)
    env_kwargs = {
        'add_risk_factor_queries': st.checkbox(
            'Add risk factor queries', value=True,
            on_change=reset_episode_state),
        'top_k_evidence': st.number_input(
            'k queries', min_value=0, value=100,
            on_change=reset_episode_state),
        'reward_type': checkpoint_args.env.value.reward_type,
        'limit_options_with_llm': st.checkbox('Limit options with LLM'),
    }
    env_kwargs['add_none_of_the_above_option'] = st.checkbox(
        'Add \"None of the Above\" option', value=True,
        on_change=reset_episode_state) \
            if env_kwargs['reward_type'] in [
                'ranking', 'continuous_dependent'] else False
    df = get_dataset(args, split)
    lmm_interface, fmm_interface = get_env_models(args)
    env = get_environment(
        args, split, df, lmm_interface, fmm_interface, **env_kwargs)
    section_size = 30
    sl = st.selectbox(
        'Dataset Section',
        [f'{i+1}-{i+section_size}'
         if i + section_size < len(df) else f'{i+1}-{len(df)}'
         for i in range(0, len(df), section_size)],
         on_change=reset_episode_state)
    valid_instances_filter = get_valid_instances_filter(
        env, args, split, sl, **env_kwargs)
    num_valid = valid_instances_filter.sum()
    if num_valid == 0:
        st.warning('No results after filtering.')
    instance_name = st.selectbox(f'Instances ({num_valid})', list(
        df[valid_instances_filter].instance_name),
        on_change=reset_episode_state)
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
    with st.spinner(
            'Extracting information from reports to set up environment...'):
        st.session_state['episode']['reset'] = env.reset(
            options=get_reset_options(args, instance_index))
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
    st.button('Jump', on_click=JumpTo(args, env, jump_to, instance_index))
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


def evidence_annotations(step_index, evidence_index):
    relevance = st.radio(
        'Is the evidence relevant?',
        ['0 - Not Useful',
         '1 - Weak Correlation',
         '2 - Useful',
         '3 - Very Useful'],
        key=f'relevant {step_index} {evidence_index}')
    impact = st.radio(
        'How does the individual impact of the evidence (the plot) align with '
        'intuition?',
        ['-1 - Opposite of expectations',
         '0 - Not aligned with expectations',
         '1 - Aligned with expectations'],
        key=f'individual impact {step_index} {evidence_index}')
    notes = st.text_area('Other notes (if needed)', key=f'notes {step_index} {evidence_index}')
    return {'relevance': relevance, 'impact': impact, 'notes': notes}


def get_evidence_distributions(actor, parameter_votes_info):
    param_votes = parameter_votes_info['param_votes']
    evidence_distributions = []
    for evidence_idx in range(param_votes.shape[1]):
        slc = slice(evidence_idx, evidence_idx + 1)
        param_info = actor.votes_to_parameters(
            parameter_votes_info['diagnosis_embeddings'],
            parameter_votes_info['context_embeddings'][slc],
            param_votes[:, slc])
        meta_dist = actor.parameters_to_dist(*param_info['params'])
        option_dist = actor.get_mean([meta_dist])[0]
        evidence_distributions.append(option_dist)
    return evidence_distributions


def get_evidence_scores(
        actor, selected_options, parameter_votes_info, evidence_distributions, i):
    # option x evidence x num_params
    param_votes = parameter_votes_info['param_votes']
    option_indices = torch.tensor(selected_options)
    if 'context_attn_weights' in parameters_info.keys() and st.checkbox(
            'Sort by attention', value=True, key=f'sort by attn {i}'):
        st.write('Ordered by attention averaged over the selected options.')
        # option x evidence
        attn = parameters_info['context_attn_weights']
        scores = attn[option_indices].mean(0)
        return scores, True
    if len(selected_options) == 1:
        st.write('Ordered by vote.')
        params = param_votes[option_indices[0]].transpose(0, 1)
        meta_dist = actor.parameters_to_dist(*params)
        scores = actor.get_mean([meta_dist])[0]
    else:
        scores = []
        # TODO: potentially do something different here
        #   if env_kwargs['reward_type'] == 'continuous_independent'
        st.write(
            'Ordered by negative entropy of the induced distribution.')
        for dist in evidence_distributions:
            entropy = Categorical(
                torch.softmax(dist[option_indices], 0)).entropy()
            scores.append(entropy)
        scores = np.array(scores)
    return scores, False


def get_evidence_strings(parameter_votes_info, parameters_info, scores, sort_by_attn):
    context_strings = parameter_votes_info['context_strings']
    context_info = parameter_votes_info['context_info']
    new_strings = []
    for cs, ci, score in zip(context_strings, context_info, scores):
        evidence = 'the report' if ci == 'report' else \
            cs if ci == 'bias' else \
            ("\"" + cs + "\"")
        print(score)
        score_string = "score={:.1f}".format(score) \
            if not sort_by_attn else \
            "attn={:.1f}%".format(score * 100)
        new_strings.append('#### ({}) {}'.format(score_string, evidence))
    return new_strings


def make_evidence_plot(options, option_dist):
    option_dist = torch.sigmoid(option_dist) \
        if env_kwargs['reward_type'] == 'continuous_independent' else \
        torch.softmax(option_dist, 0)
    # options_scores = sorted(zip(options, option_dist), key=lambda x: -x[1])
    # for o, od in options_scores:
    #     st.write('- ({:.1f}%) {}'.format(od * 100, o))
    data = pd.DataFrame({
        'option': [o.split('(')[0].strip() for o in options],
        'probability': option_dist,
    })
    # st.bar_chart(data, x='option', y='probability')
    fig, ax = plt.subplots(1, 1, figsize=(3, len(data) * .6))
    chart = sns.barplot(data, x='option', y='probability', ax=ax)
    chart.set_ylim(0, 1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 20)
    for i, v in enumerate(option_dist):
        ax.text(i - .1, v.item() + 0.01, '{:.1f}%'.format(v.item() * 100))
    st.pyplot(fig)


def display_action_evidence(
        args, actor_checkpoint, i, parameter_votes_info, parameters_info,
        options, selected_options=None):
    st.write('### Individual Evidence Impact')
    actor = get_actor(args, actor_checkpoint)
    if isinstance(actor.observation_embedder, BertObservationEmbedder):
        st.write("Not an Interpretable model, nothing to display.")
        return
    if selected_options is not None and len(selected_options) > 0:
        selected_options = sorted(selected_options)
        selected_option_strings = [options[idx] for idx in selected_options]
        st.write('Ordered with respect to "{}".'.format(
            '", "'.join(selected_option_strings)))
    else:
        st.warning('Some options must be selected to order by.')
        return
    with torch.no_grad():
        evidence_dists = get_evidence_distributions(
            actor, parameter_votes_info)
        scores, sort_by_attn = get_evidence_scores(
            actor, selected_options, parameter_votes_info, evidence_dists, i)
    evidence_strings = get_evidence_strings(
        parameter_votes_info, parameters_info, scores, sort_by_attn)
    evidence_info = sorted(zip(
        range(len(evidence_strings)),
        evidence_strings,
        evidence_dists,
        scores), key=lambda x: x[-1])
    anns = {}
    for j, es, ed, _ in evidence_info:
        c1, c2 = st.columns([1, 3])
        with c1:
            make_evidence_plot(
                selected_option_strings, ed[torch.tensor(selected_options)])
        with c2:
            st.write(es)
            if annotate:
                anns[j] = evidence_annotations(i, j)
    return anns


def annotate_timepoint(i):
    st.write('### Timepoint annotations')
    st.write('Given the notes and evidence seen so far, which of the following are correct diagnoses?')
    selected_options = set()
    diagnosis_salience = {}
    for option in options:
        if st.checkbox(option, key=f'correct {option} {i}'):
            selected_options.add(option)
            diagnosis_salience[option] = st.radio('Select the Salience of the diagnosis.', ['Primary', 'Secondary'], key=f'diagnosis salience {i} {option}')
    temporal_mistakes = selected_options.intersection(info['current_targets'])
    extraction_mistakes = selected_options.difference(info['current_targets'])
    confident_diagnoses_not_in_differentials = st.text_area('Please write a comma-separated list of any confident diagnoses that are not listed.', key=f'text1 {i}')
    confident_diagnoses_not_in_differentials = [x.strip() for x in confident_diagnoses_not_in_differentials.split(',')]
    missed_differentials = st.text_area('Please write a comma-separated list of any differential diagnoses that are not listed but should be considered.', key=f'text2 {i}')
    missed_differentials = [x.strip() for x in missed_differentials.split(',')]
    if len(temporal_mistakes) > 0:
        st.radio(f'The following were not extracted as \"confident\" until later in time: {temporal_mistakes}. Is this because the clinician may have made a mistake?', ['Yes', 'No'], key=f'temporal mistake {i}')
    if len(extraction_mistakes) > 0:
        st.write(f'The following were not extracted as confident diagnoses at all: {extraction_mistakes}. These represent mistakes by the confident diagnosis extraction model.')


def display_action(options, action, action_stddev, i):
    if env_kwargs['reward_type'] == 'continuous_independent':
        ratings = torch.sigmoid(torch.tensor(action))
        st.write('This actor\'s scores are interpreted as independent. '
                 '(Probabilities do not sum to 1.)')
    else:
        ratings = torch.softmax(torch.tensor(action), 0).numpy()
        st.write('This actor\'s scores are interpreted as not independent. '
                 '(Probabilities sum to 1.)')
    options_ratings_indices = list(
        zip(options, ratings, action_stddev, range(len(action))))
    options_ratings_indices = sorted(
        options_ratings_indices, key=lambda x: -x[1])
    selected_options = set()
    for o, r, a_s, j in options_ratings_indices:
        option_string ='({:.1f}%{}) {}'.format(
            r * 100, '±{:.1f}'.formate(a_s * 100) if a_s != 0 else '', o)
        if st.checkbox(option_string, value=True,
                key=f'option checkbox {i} {option_string}'):
            selected_options.add(j)
    st.write('Select options above to use to sort the evidence.')
    return selected_options


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
            if annotate:
                annotate_timepoint(i)
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
                parameter_votes_info, parameters_info, action, \
                    action_stddev = sample_action(
                    args, observation, actor_checkpoint)
                action = np.array(action, dtype=float)
                selected_options = display_action(
                    options, action, action_stddev, i)
                anns = display_action_evidence(
                    args, actor_checkpoint, i, parameter_votes_info,
                    parameters_info, options,
                    selected_options=selected_options)
            else:
                raise Exception
            action_submitted = st.button('Submit Action', key=f'submit {i}')
            skip_past = st.session_state['episode']['skip_to'] is not None and \
                len(st.session_state['episode']['steps']) + \
                1 < st.session_state['episode']['skip_to']
            if not action_submitted and not skip_past and not (
                    not observation['evidence_is_retrieved'] and args['skip_query']):
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
