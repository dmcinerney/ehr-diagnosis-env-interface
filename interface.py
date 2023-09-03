from utils import *
import streamlit as st
import pandas as pd
import numpy as np
from ehr_diagnosis_agent.models.actor import InterpretableDirichletActor, \
    InterpretableNormalActor, InterpretableBetaActor, InterpretableDeltaActor
from ehr_diagnosis_agent.models.observation_embedder import \
    BertObservationEmbedder
import torch
from torch.distributions import Categorical, Bernoulli, kl_divergence
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
from time import sleep
import pickle as pkl
import random
from collections import Counter
from torch import nn


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
    split_ckpt_path = args['models'][actor_checkpoint].split('/')
    config_path = '/'.join(split_ckpt_path[:-1] + ['config.yaml'])
    return OmegaConf.load(config_path)


@st.cache_resource
def get_actor(args, actor_checkpoint):
    checkpoint_args = get_checkpoint_args(args, actor_checkpoint)
    actor_params = checkpoint_args.actor.value['{}_params'.format(
        checkpoint_args.actor.value.type)]
    actor_params.update(checkpoint_args.actor.value['shared_params'])
    actor = actor_types[checkpoint_args.actor.value.type](actor_params)
    state_dict = torch.load(args['models'][actor_checkpoint])['actor']
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


@st.cache_resource
def get_instance_metadata_with_model_output_metadata(
        _env, outputs_to_add, args, split):
    df = get_instance_metadata(_env, args, split)
    if outputs_to_add != 'No model selected':
        model_outputs = pd.read_csv(
            args['model_outputs'][split][outputs_to_add])
        df = df.merge(model_outputs, how='left', on='episode_idx')
    return df


def make_plot(options, probs, not_dist=False, ylabel='probability', ax=None):
    data = pd.DataFrame({
        'option': options,
        ylabel: probs,
    })
    # st.bar_chart(data, x='option', y='probability')
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
        fig = None
    chart = sns.barplot(data, x='option', y=ylabel, ax=ax)
    if not not_dist:
        chart.set_ylim(0, 1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 20)
    for i, v in enumerate(probs):
        ax.text(i - .2, v.item() + 0.01, '{:.1f}%'.format(v.item() * 100))
    if fig is not None:
        st.pyplot(fig)


st.set_page_config(layout="wide")
st.title('EHR Diagnosis Environment Visualizer')
args = get_args('config.yaml')
dataset = args['data']['dataset']
st.write(f'Dataset: \"{dataset}\"')
splits = get_splits(args)
with st.sidebar:
    annotate = st.checkbox('Annotate')
    show_state = True
    if annotate:
        if st.button('Restart Report Timer'):
            if 'display_report_timer' in st.session_state.keys():
                del st.session_state['display_report_timer']
        show_state = st.checkbox('Show state')
        num_evidence_to_annotate = st.number_input(
            'Number of evidence snippets to annotate', min_value=1, value=10)
        show_remaining_anns = st.checkbox('Show remaining evidence')
    show_instance_metadata = st.checkbox('Show instance metadata')
    ignore_selected_evidence = st.checkbox('Ignore selected evidence types')
    options = list(args['models'].keys())
    if not annotate:
        options = ['Choose your own actions'] + options
    actor_checkpoint = st.selectbox('Pick an actor', options)
with st.sidebar:
    st.write('#### Environment')
    use_random_start_idx = st.checkbox(
        'Start at a random timestep', on_change=reset_episode_state)
    reward_type = args['env']['reward_type']
    if actor_checkpoint != 'Choose your own actions':
        checkpoint_args = get_checkpoint_args(args, actor_checkpoint)
        st.write(f'Reward type: {checkpoint_args.env.value.reward_type}')
        assert reward_type == checkpoint_args.env.value.reward_type
    split = st.selectbox(
        'Dataset Split', splits,
        index=splits.index('val1') if 'val1' in splits else 0,
        on_change=reset_episode_state)
    df = get_dataset(args, split)
    llm_interface, fmm_interface = get_env_models(args)
    env = get_environment(
        args, split, df, llm_interface, fmm_interface)
    st.write('#### Instances')
    options = ['No model selected'] + list(
        args['model_outputs'][split].keys())
    outputs_to_add = st.selectbox(
        'Add pre-computed model outputs to the metadata by selecting a model.',
        options)
    instance_metadata = get_instance_metadata_with_model_output_metadata(
        env, outputs_to_add, args, split)
    filter_instances_string = st.text_input(
        'Type a lambda expression in python that filters instances using their'
        ' cached metadata.')
    if filter_instances_string == '':
        filtered_instance_metadata = instance_metadata
    else:
        filtered_instance_metadata = instance_metadata[instance_metadata.apply(
            eval(filter_instances_string), axis=1)]
    valid_instances = filtered_instance_metadata.episode_idx
    num_valid = len(valid_instances)
    if num_valid == 0:
        st.warning('No results after filtering.')
        st.stop()
    st.divider()
    st.write('Configuration')
    st.write(args)
    st.divider()
    spinner_container = st.container()
if show_instance_metadata:
    with st.expander('Instance metadata'):
        st.write(filtered_instance_metadata[:100])
        st.write('...')
        if 'all_targets' in filtered_instance_metadata.columns:
            counts = Counter()
            rows = filtered_instance_metadata[
                ~filtered_instance_metadata.all_targets.isna()]
            for i, row in rows.iterrows():
                for x in eval(row.all_targets):
                    counts[x] += 1
            # st.write(counts)
            options = sorted(list(counts.keys()))
            probs = np.array([counts[o] for o in options]) / len(rows)
            with st.columns([2, 5])[0]:
                make_plot(options, probs)
if annotate:
    annotator_name = st.text_input('Annotator Name')
    if annotator_name == '':
        st.warning(
            'You need to specify an annotator name to submit annotations.')
        st.stop()
instance_name = st.selectbox(f'Instances ({num_valid})', list(
    df.iloc[valid_instances].instance_name),
    on_change=reset_episode_state,
    format_func=(lambda x: x) if not annotate else
        lambda x: ' '.join(x.split()[:2]))
if 'episode' in st.session_state.keys() and \
        instance_name != st.session_state['episode']['instance']:
    reset_episode_state()
if not annotate:
    st.button('Reset session', on_click=reset_episode_state)
    jump_timestep_container = st.container()
if num_valid == 0:
    st.warning('No example to show')
    st.stop()
# TODO: info exists in metadata, compute class prevelances
# if os.path.exists('class_prevelances.pkl'):
#     with open('class_prevelances.pkl', 'rb') as f:
#         class_prevelances = pkl.load(f)
# else:
#     class_prevelances = None
#     st.write('To allow sorting evidence based on the difference with class '
#              'prevelance, you need to postprocess the instance metadata in the'
#              ' sidebar.')
container = st.empty()

# st.write(str(env.model.model.lm_head.weight.device))
instance_index = np.where(df.instance_name == instance_name)[0].item()
if 'episode' not in st.session_state.keys():
    st.session_state['episode'] = {'instance': instance_name}
    # store the return value of reset for the current instance
    with st.spinner(
            'Extracting information from reports to set up environment...'):
        st.session_state['episode']['reset'] = env.reset(
            options=get_reset_options(args, instance_index))
        if use_random_start_idx:
            info = st.session_state['episode']['reset'][1]
            random_start_idx = random.randint(
                info['current_report'],
                info['current_report'] + info['max_timesteps'] // 2 - 1)
            st.session_state['episode']['reset'] = env.reset(
                options=get_reset_options(
                    args, instance_index, start_report_index=random_start_idx))
    # store the actions for the current instance
    st.session_state['episode']['actions'] = {}
    # store the return value of the step function for the current instance
    st.session_state['episode']['steps'] = {}
    st.session_state['episode']['skip_to'] = None
observation, info = st.session_state['episode']['reset']
if not annotate:
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
    if not annotate:
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
        st.write('**max_future_reports**: {}'.format(info['max_timesteps'] // 2))
        for k, v in info.items():
            if k in ['current_report', 'past_reports', 'future_reports', 'current_targets', 'max_timesteps']:
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
    impact = None
    if not relevance.startswith('0'):
        impact = st.radio(
            'How does the individual impact of the evidence (the plot) align with '
            'intuition?',
            ['-1 - Opposite of expectations',
            '0 - Not aligned with expectations',
            '1 - Aligned with expectations'],
            key=f'individual impact {step_index} {evidence_index}')
    notes = st.text_area('Other notes (if needed)', key=f'evidence notes {step_index} {evidence_index}')
    return {'relevance': relevance, 'impact': impact, 'notes': notes}


def get_evidence_dist(actor, parameter_votes_info, evidence_idx):
    if actor.has_bias and evidence_idx > 0:
        # first piece of context is always the bias vector
        slc = torch.tensor([0, evidence_idx])
    else:
        slc = torch.tensor([evidence_idx])
    param_info = actor.votes_to_parameters(
        parameter_votes_info['diagnosis_embeddings'],
        parameter_votes_info['context_embeddings'][slc],
        parameter_votes_info['param_votes'][:, slc])
    meta_dist = actor.parameters_to_dist(*param_info['params'])
    return actor.get_mean([meta_dist])[0]


def get_evidence_distributions(actor, parameter_votes_info):
    evidence_distributions = []
    for evidence_idx in range(parameter_votes_info['param_votes'].shape[1]):
        evidence_distributions.append(get_evidence_dist(
            actor, parameter_votes_info, evidence_idx))
    return evidence_distributions


def get_evidence_scores(
        actor, action, options, selected_options, parameter_votes_info, evidence_distributions, i):
    # option x evidence x num_params
    param_votes = parameter_votes_info['param_votes']
    option_indices = torch.tensor(selected_options)
    sort_options = []
    if 'context_attn_weights' in parameters_info.keys():
        sort_options += ['Sort by attention']
    if reward_type in ['continuous_dependent', 'ranking']:
        sort_options += ['Sort by the denominator of the softmax']
    if len(selected_options) >= 2:
        if "[bias vector]" in parameter_votes_info['context_strings']:
            sort_options += [
                'Sort by MSE with the bias logits',
                'Sort by the symmetric KL divergence with the bias distribution',
                'Sort by the symmetric KL divergence with the agg/bias distribution',
            ]
        # peakiness options
        sort_options += [
            'Sort by the entropy of the induced distributions',
            'Sort by the maximum probability',
        ]
    for selected_option in selected_options:
        sort_options += [f'Sort by the probability of \"{options[selected_option]}\"']
    # if annotate:
    #     sort_options = sort_options[:1]
    bias_idx = parameter_votes_info['context_strings'].index(
        "[bias vector]")
    agg_over_bias_logits = torch.tensor(action[option_indices]) - \
        evidence_distributions[bias_idx][option_indices]
    # if not annotate:
    #     c1, c2, _ = st.expander('bias and agg/bias distributions').columns(
    #         [2, 2, 1])
    #     with c1:
    #         st.write("bias dist")
    #         make_evidence_plot(
    #             [options[idx] for idx in option_indices],
    #             evidence_distributions[bias_idx][option_indices])
    #     with c2:
    #         st.write("agg/bias dist")
    #         # make_evidence_plot(
    #         #     [options[idx] for idx in option_indices], agg_over_bias_logits)
    #         make_plot(
    #             [options[idx] for idx in option_indices],
    #             torch.sigmoid(torch.tensor(action[option_indices])) / torch.sigmoid(evidence_distributions[bias_idx][option_indices])
    #             if env_kwargs['reward_type'] == 'continuous_independent' else
    #             torch.softmax(torch.tensor(action[option_indices]), 0) / torch.softmax(evidence_distributions[bias_idx][option_indices], 0),
    #             not_dist=True,
    #         )
    if annotate:
        sort_by = sort_options[0]
    else:
        sort_by = st.radio(
            'Choose how to sort the evidence.', sort_options, key=f'sort by {i}')
    if sort_by.startswith('Sort by the probability of'):
        option_to_sort_by = options.index(sort_by.split('\"')[1])
        selected_option_to_sort_by = selected_options.index(option_to_sort_by)
        # params = param_votes[option_to_sort_by].transpose(0, 1)
        # meta_dist = actor.parameters_to_dist(*params)
        # scores = actor.get_mean([meta_dist])[0]
        scores = []
        for dist in evidence_distributions:
            if reward_type == 'continuous_independent':
                scores.append(torch.sigmoid(
                    dist[option_indices])[selected_option_to_sort_by])
            else:
                scores.append(torch.softmax(
                    dist[option_indices], 0)[selected_option_to_sort_by])
        ascending = False
        score_format = lambda x: 'Prob={:.1f}%'.format(x * 100)
    elif sort_by == 'Sort by attention':
        # option x evidence
        attn = parameters_info['context_attn_weights']
        scores = attn[option_indices].mean(0)
        ascending = False
        score_format = lambda x: 'Attn={:.1f}%'.format(x * 100)
    elif sort_by == 'Sort by the denominator of the softmax':
        scores = []
        for dist in evidence_distributions:
            scores.append(torch.exp(dist).sum())
        ascending = False
        score_format = lambda x: 'Softmax denom={:.1f}'.format(x)
    elif sort_by == 'Sort by the entropy of the induced distributions':
        scores = []
        # TODO: potentially do something different here
        #   if env_kwargs['reward_type'] == 'continuous_independent'
        for dist in evidence_distributions:
            entropy = Categorical(
                torch.softmax(dist[option_indices], 0)).entropy()
            scores.append(entropy)
        scores = np.array(scores)
        ascending = True
        score_format = lambda x: 'Entropy={:.1f}'.format(x)
    elif sort_by == 'Sort by the maximum probability':
        scores = []
        for dist in evidence_distributions:
            if reward_type == 'continuous_independent':
                scores.append(torch.sigmoid(dist[option_indices]).max())
            else:
                scores.append(torch.softmax(dist[option_indices], 0).max())
        scores = np.array(scores)
        ascending = False
        score_format = lambda x: 'Max Prob={:.1f}%'.format(x * 100)
    elif sort_by == 'Sort by MSE with the bias logits':
        scores = []
        bias_logits = evidence_distributions[bias_idx][option_indices]
        for dist in evidence_distributions:
            scores.append(nn.MSELoss()(
                dist[option_indices], bias_logits).item())
        scores = np.array(scores)
        ascending = False
        score_format = lambda x: 'MSE with Bias Logits={:.1f}'.format(x)
    elif sort_by == 'Sort by the symmetric KL divergence with the bias ' \
            'distribution':
        scores = []
        if reward_type == 'continuous_independent':
            bias_distribution = Bernoulli(
                torch.sigmoid(evidence_distributions[bias_idx][option_indices]))
        else:
            bias_distribution = Categorical(
                torch.softmax(evidence_distributions[bias_idx][option_indices], 0))
        for dist in evidence_distributions:
            if reward_type == 'continuous_independent':
                dist = Bernoulli(
                    torch.sigmoid(dist[option_indices]))
            else:
                dist = Categorical(
                    torch.softmax(dist[option_indices], 0))
            scores.append((
                kl_divergence(dist, bias_distribution) +
                kl_divergence(bias_distribution, dist)).mean() / 2)
        scores = np.array(scores)
        ascending = False
        score_format = lambda x: 'Sym KL with Bias={:.1f}'.format(x)
    elif sort_by == 'Sort by the symmetric KL divergence with the agg/bias ' \
            'distribution':
        scores = []
        if reward_type == 'continuous_independent':
            agg_over_bias_distribution = Bernoulli(
                torch.sigmoid(agg_over_bias_logits))
        else:
            agg_over_bias_distribution = Categorical(
                torch.softmax(agg_over_bias_logits, 0))
        for dist in evidence_distributions:
            if reward_type == 'continuous_independent':
                dist = Bernoulli(
                    torch.sigmoid(dist[option_indices]))
            else:
                dist = Categorical(
                    torch.softmax(dist[option_indices], 0))
            scores.append((
                kl_divergence(dist, agg_over_bias_distribution) +
                kl_divergence(agg_over_bias_distribution, dist)).mean() / 2)
        scores = np.array(scores)
        ascending = True
        score_format = lambda x: 'Sym KL with Agg/Bias={:.2f}'.format(x)
    else:
        raise Exception
    return sort_by, ascending, scores, score_format


def get_evidence_strings(parameter_votes_info, parameters_info, scores, score_format):
    context_strings = parameter_votes_info['context_strings']
    context_info = parameter_votes_info['context_info']
    new_strings = []
    for cs, ci, score in zip(context_strings, context_info, scores):
        evidence = '[report]' if ci == 'report' else cs
        score_string = score_format(score)
        new_strings.append({
            'score': score_string.split('=')[1],
            'evidence': evidence,
        })
        if ci.startswith('evidence'):
            split_evidence = evidence.split(':')
            query = split_evidence[0]
            the_rest = ':'.join(split_evidence[1:])
            new_evidence, day = the_rest.split(' (day ')
            new_strings[-1]['evidence'] = new_evidence.strip()
            new_strings[-1]['query'] = query
            new_strings[-1]['day'] = int(float(day[:-1])) \
                if day[:-1] != 'nan' else 'UNK'
            new_strings[-1]['report_number'] = ci.split(' (report ')[1][:-1]
    return new_strings


def make_evidence_plot(options, option_dist, bias_dist=None):
    option_dist = torch.sigmoid(option_dist) \
        if reward_type == 'continuous_independent' else \
        torch.softmax(option_dist, 0)
    options = [o.split('(')[0].strip() for o in options]
    if bias_dist is not None:
        fig, axs = plt.subplots(2, 1, figsize=(3, 4), sharex=True)
        ax0 = axs[0]
        ax1 = axs[1]
    else:
        ax0 = None
    make_plot(options, option_dist, ax=ax0)
    if bias_dist is not None:
        bias_dist = torch.sigmoid(bias_dist) \
            if reward_type == 'continuous_independent' else \
            torch.softmax(bias_dist, 0)
        make_plot(options, option_dist / bias_dist, not_dist=True,
                  ylabel='Risk over Baseline', ax=ax1)
        ax1.plot([-.5, -.5 + len(options)], [1.0, 1.0], '--', color='black')
        ax1.set_xlim([-.5,  -.5 + len(options)])
        st.pyplot(fig)


def display_evidence_string(es):
    # needs (background, key, value)
    tag_html_template = \
        '<span><span style="display:inline-flex;flex-direction:row;align-ite' \
        'ms:center;background:{};border-radius:0.5rem;padding:0.25rem 0.5rem' \
        ';overflow:hidden;line-height:1"><span style="margin-right:0.5rem;fo' \
        'nt-size:0.75rem;opacity:0.5">{}</span><span style="border-right:1px' \
        ' solid;opacity:0.1;margin-right:0.5rem;align-self:stretch"></span>{' \
        '}</span></span>'
    score = es['score']
    evidence = es['evidence']
    tag_info = [('#803df533', 'SCORE', score)]
    if 'query' in es.keys():
        tag_info += [
            ('#803df533', 'QUERY', es['query']),
            ('#803df533', 'REPORT NUMBER', es['report_number']),
            ('#803df533', 'DAY', es['day']),
        ]
    st.write(
        ' '.join([
            tag_html_template.format(background, key, value)
            for background, key, value in tag_info
        ]),
        unsafe_allow_html=True
    )
    st.write('#### ' + evidence)


def display_action_evidence(
        args, actor_checkpoint, i, action, parameter_votes_info, parameters_info,
        options, selected_options=None):
    actor = get_actor(args, actor_checkpoint)
    if isinstance(actor.observation_embedder, BertObservationEmbedder):
        st.write("Not an Interpretable model, nothing to display.")
        return {}
    if selected_options is not None and len(selected_options) > 0:
        selected_options = sorted(selected_options)
        selected_option_strings = [options[idx] for idx in selected_options]
    else:
        st.warning('Some options must be selected to order by.')
        return {}
    with torch.no_grad():
        evidence_dists = get_evidence_distributions(
            actor, parameter_votes_info)
        sort_by, ascending, scores, score_format = get_evidence_scores(
            actor, action, options, selected_options, parameter_votes_info,
            evidence_dists, i)
    evidence_strings = get_evidence_strings(
        parameter_votes_info, parameters_info, scores, score_format)
    if actor.has_bias:
        assert evidence_strings[0]['evidence'] == '[bias vector]'
        bias_dist = evidence_dists[0]
    else:
        bias_dist = None
    evidence_info = sorted(zip(
        range(len(evidence_strings)),
        evidence_strings,
        evidence_dists,
        scores), key=lambda x: x[-1] if ascending else -x[-1])
    evidence_info_to_show = evidence_info
    ignore_evidence_types = set([])
    if ignore_selected_evidence:
        st.write('Ignore evidence of:')
        evidence_types = sorted(list(set(
            [es['query']
            for _, es, _, _ in evidence_info_to_show if 'query' in es.keys()])))
        num_columns = 3
        columns = st.columns(num_columns)
        for idx, x in enumerate(evidence_types):
            with columns[idx % num_columns]:
                if st.checkbox(x):
                    ignore_evidence_types.add(x)
        evidence_info_to_show = [
            x for x in evidence_info_to_show
            if 'query' not in x[1].keys() or
            x[1]['query'] not in ignore_evidence_types]
    if annotate:
        anns = {'evidence_anns': {}}
        evidence_info_to_show = evidence_info_to_show[:num_evidence_to_annotate]
        st.write(f'Only showing top {num_evidence_to_annotate} pieces of evidence.')
    else:
        anns = {}
    for j, es, ed, _ in evidence_info_to_show:
        c1, c2 = st.columns([1, 3])
        with c1:
            make_evidence_plot(
                selected_option_strings, ed[torch.tensor(selected_options)],
                bias_dist=bias_dist[torch.tensor(selected_options)]
                    if bias_dist is not None else None)
        with c2:
            display_evidence_string(es)
            if annotate:
                anns['evidence_anns'][j] = evidence_annotations(i, j)
                anns['evidence_anns'][j]['evidence'] = es
                anns['evidence_anns'][j]['evidence_distribution'] = ed.tolist()
    if annotate and show_remaining_anns:
        if len(evidence_info) > num_evidence_to_annotate:
            with st.expander('The rest of the evidence.'):
                for j, es, ed, _ in evidence_info[num_evidence_to_annotate:]:
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        make_evidence_plot(
                            selected_option_strings,
                            ed[torch.tensor(selected_options)],
                            bias_dist=bias_dist[torch.tensor(selected_options)]
                                if bias_dist is not None else None)
                        
                    with c2:
                        display_evidence_string(es)
            anns['options'] = options
            anns['selected_options'] = selected_options
            anns['sort_by'] = sort_by
        else:
            st.write('No more evidence.')
    return anns


def annotate_timepoint(i, info):
    timepoint_anns = {}
    st.write('Select any illnesses with which the patient has already been '
             'confidently diagnosed:')
    seen_targets = set()
    for option in options:
        if st.checkbox(option, key=f'already seen {option} {i}'):
            seen_targets.add(option)
    timepoint_anns['seen_targets'] = seen_targets
    if len(seen_targets) == 0:
        timepoint_anns['option_likelihood_anns'] = {}
        for option in options:
            timepoint_anns['option_likelihood_anns'][option] = st.radio(
                f'What is the likelihood of the patient having {option}?',
                ['Unlikely', 'Somewhat likely', 'Very likely'],
                key=f'likelihood ann {i} {option}')
    else:
        timepoint_anns['invalid_instance_notes'] = st.text_area('Other notes (if needed)', key=f'invalid instance notes {i}')
    # st.write('Given the notes and evidence seen so far, which of the following are correct diagnoses?')
    # selected_options = set()
    # diagnosis_salience = {}
    # for option in options:
    #     if st.checkbox(option, key=f'correct {option} {i}'):
    #         selected_options.add(option)
    #         diagnosis_salience[option] = st.radio('Select the Salience of the diagnosis.', ['Primary', 'Secondary'], key=f'diagnosis salience {i} {option}')
    # temporal_mistakes = selected_options.intersection(info['current_targets'])
    # extraction_mistakes = selected_options.difference(info['current_targets'])
    # confident_diagnoses_not_in_differentials = st.text_area('Please write a comma-separated list of any confident diagnoses that are not listed.', key=f'text1 {i}')
    # confident_diagnoses_not_in_differentials = [x.strip() for x in confident_diagnoses_not_in_differentials.split(',')]
    # missed_differentials = st.text_area('Please write a comma-separated list of any differential diagnoses that are not listed but should be considered.', key=f'text2 {i}')
    # missed_differentials = [x.strip() for x in missed_differentials.split(',')]
    # if len(temporal_mistakes) > 0:
    #     st.radio(f'The following were not extracted as \"confident\" until later in time: {temporal_mistakes}. Is this because the clinician may have made a mistake?', ['Yes', 'No'], key=f'temporal mistake {i}')
    # if len(extraction_mistakes) > 0:
    #     st.write(f'The following were not extracted as confident diagnoses at all: {extraction_mistakes}. These represent mistakes by the confident diagnosis extraction model.')
    return timepoint_anns


def annotate_timepoint_end(i, info, anns):
    timepoint_anns = {}
    timepoint_anns['option_likelihood_anns2'] = {}
    st.write("Review your previous answers to the following questions. If you feel your opinion has changed, change your answers accordingly.")
    answers = ['Unlikely', 'Somewhat likely', 'Very likely']
    for option in options:
        timepoint_anns['option_likelihood_anns2'][option] = st.radio(
            f'What is the likelihood of the patient having {option}?',
            answers,
            index=answers.index(anns['option_likelihood_anns'][option]),
            key=f'likelihood ann 2 {i} {option}')
    timepoint_anns['concluding_notes'] = st.text_area('Other notes (if needed)', key=f'concluding notes {i}')
    # timepoint_anns['prediction_ann'] = st.radio('Does the prediction align with your expert intuition?', ['Yes', 'No'], key=f'prediction ann {i}')
    # timepoint_anns['change_opinion'] = st.radio('Does the evidence change your opinion?', ['Yes', 'No'], key=f'change opinion ann {i}')
    return timepoint_anns


def display_action(args, options, action, action_stddev, i, actor_checkpoint, parameter_votes_info):
    c1, c2 = st.columns([2, 5])
    anns = {}
    with c2: 
        if reward_type == 'continuous_independent':
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
        if annotate:
            selected_options = set([
                j for _, _, _, j in options_ratings_indices])
            anns['prediction_aligns'] = st.radio(
                'Does this prediction align with your likelihood intuition.',
                ['Yes', 'No'], key=f'prediction aligns {i}')
        else:
            selected_options = set()
            for o, r, a_s, j in options_ratings_indices:
                option_string ='({:.1f}%{}) {}'.format(
                    r * 100, '±{:.1f}'.format(a_s * 100) if a_s != 0 else '', o)
                if st.checkbox(option_string, value=True,
                        key=f'option checkbox {i} {option_string}'):
                    selected_options.add(j)
            st.write('Select options above to use to sort the evidence.')
    with c1:
        actor = get_actor(args, actor_checkpoint)
        if actor.has_bias:
            bias_dist = get_evidence_dist(actor, parameter_votes_info, 0)
            bias_dist = bias_dist[torch.tensor(sorted(selected_options))]
        else:
            bias_dist = None
        make_evidence_plot(
            [options[j] for j in selected_options],
            torch.tensor(action)[torch.tensor(sorted(selected_options))],
            bias_dist=bias_dist)
    return selected_options, anns


def write_annotations(
        split, instance_name, i, observation, info, reports, anns):
    all_anns = {
        'info': {
            k: v for k, v in info.items()
            if k not in ['past_reports', 'future_reports']},
        'obs': {
            k: v for k, v in observation.items()
            if k not in ['reports', 'evidence']},
        'timestep': i,
        'num_reports': len(reports),
        'instance': f'{split} {instance_name}',
        **anns
    }
    # st.success(all_anns)
    if not os.path.exists(args['annotations']['path']):
        os.mkdir(args['annotations']['path'])
    ann_path = os.path.join(
        args['annotations']['path'], '_'.join(annotator_name.split()))
    if not os.path.exists(ann_path):
        os.mkdir(ann_path)
    next_idx = 0
    for filename in os.listdir(ann_path):
        if filename.startswith('ann_') and filename.endswith('.pkl'):
            next_idx = max(
                next_idx, int(filename.split('.')[0].split('_')[1]) + 1)
    new_filepath = os.path.join(ann_path, f'ann_{next_idx}.pkl')
    st.success(f'Writing to: {new_filepath}')
    with open(new_filepath, 'wb') as f:
        pkl.dump({next_idx: all_anns}, f)


def get_current_annotations():
    ann_path = os.path.join(
        args['annotations']['path'], '_'.join(annotator_name.split()))
    if not os.path.exists(ann_path):
        return pd.DataFrame([])
    df = {}
    for filename in os.listdir(ann_path):
        if filename.startswith('ann_') and filename.endswith('.pkl'):
            with open(os.path.join(ann_path, filename), 'rb') as f:
                df.update(pkl.load(f))
    return pd.DataFrame(df).transpose()


import time
class Timer:
    def __init__(self, num_seconds):
        self.num_total_seconds = num_seconds
        self.time_started = time.time()

    def get_seconds_left(self):
        return self.num_total_seconds - (time.time() - self.time_started)


def display_report(reports_df, key):
    if len(reports_df) == 0:
        st.write('No reports to display.')
        return
    report_names = reports_df.report_name
    def write_report(key_addon):
        report_name = st.selectbox(
            'Choose a report', report_names,
            key=key + ' ' + key_addon, index=len(report_names) - 1)
        report_row = reports_df[reports_df.report_name == report_name].iloc[0]
        st.write(f'Description: {report_row.description}')
        st.divider()
        st.write(report_row.text)
        # tabs = st.tabs([x[:10] for x in report_names])
        # for tab, report_name in zip(tabs, report_names):
        #     report_row = reports_df[reports_df.report_name == report_name].iloc[0]
        #     with tab:
        #         st.write(f'Description: {report_row.description}')
        #         st.divider()
        #         st.write(report_row.text)
    if annotate:
        if 'display_report_timer' in st.session_state.keys() and st.session_state['display_report_timer'] is None:
            # st.write("Reports have disappeared!")
            with st.expander("Reports"):
                write_report('1')
            return
        if 'display_report_timer' not in st.session_state.keys() or \
                st.session_state['display_report_timer'].get_seconds_left() <= 0:
            st.session_state['display_report_timer'] = Timer(120)
    container = st.empty()
    with container.container():
        write_report('2')
    if annotate:
        timer_context = st.empty()
        button_context = st.empty()
        if not button_context.button('Done with reports'):
            with timer_context:
                while st.session_state['display_report_timer'] is not None:
                    time.sleep(st.session_state['display_report_timer'].get_seconds_left() % 1)
                    seconds_left = st.session_state['display_report_timer'].get_seconds_left()
                    if seconds_left < 0:
                        seconds_left = 0
                    st.warning('{:.0f} seconds left!'.format(seconds_left))
                    if seconds_left <= 0:
                        st.session_state['display_report_timer'] = None
        else:
            st.session_state['display_report_timer'] = None
        button_context.empty()
        container.empty()
        # container.write("Reports have disappeared!")
        with container.expander("Reports"):
            write_report('3')


num_steps_visualized = 0
while not (terminated or truncated):
    if i not in st.session_state['episode']['steps'].keys():
        if num_steps_visualized > 0:
            container.empty()
            sleep(0.01)
        num_steps_visualized += 1
        show = observation['evidence_is_retrieved'] or not args['skip_query']
        anns = {}
        with container.container():
            if not show:
                st.write('')
            if show:
                if show_state:
                    display_state(observation, info, reward)
                st.subheader("Reports")
                reports = pd.concat(
                    [info['past_reports'],
                     df_from_string(observation['reports'])]).reset_index()
                reports['date'] = pd.to_datetime(reports['date'])
                reports = process_reports(reports)
                # reports['report_name'] = [
                #     ('(past) ' if j < len(info['past_reports']) else '')
                #     + report_name
                #     for j, report_name in enumerate(reports['report_name'])]
                display_report(
                    reports,
                    f'timestep {i + 1}')
                if not annotate:
                    st.subheader("Evidence")
                    if observation['evidence'].strip() != '':
                        st.write(df_from_string(observation['evidence']))
                    else:
                        st.write('No evidence yet!')
                if annotate:
                    st.write('### Annotations')
                    part1_container, prediction_container, evidence_container, part4_container = st.tabs([
                        'Part 1: Likelihood',
                        'Part 2: Model Prediction',
                        'Part 3: Evidence',
                        'Part 4: Concluding Thoughts'
                    ])
                else:
                    prediction_container, evidence_container = st.tabs(['Model Prediction', 'Evidence'])
                options = df_from_string(observation['options']).apply(
                    lambda r: f'{r.option} ({r.type})', axis=1).to_list()
                if annotate:
                    with part1_container:
                        anns.update(annotate_timepoint(i, info))
                # if not annotate or len(anns['seen_targets']) == 0:
                #     st.subheader("Prediction")
            if actor_checkpoint == 'Choose your own actions':
                action = [0] * len(options)
                if show:
                    with prediction_container:
                        st.write('Rank the options.')
                        action = np.array(action, dtype=float)
                        action_df = pd.DataFrame(
                            {'rating': {d: a for d, a in zip(options, action)}})
                        edited_df = st.experimental_data_editor(
                            action_df, key=f'action editor {i}')
                        # edited_df = st.data_editor(action_df, key=f'action editor {i}')
                        action_dict = edited_df.to_dict()['rating']
                        action = [action_dict[d] for d in options]
            else:
                parameter_votes_info, parameters_info, action, \
                    action_stddev = sample_action(
                    args, observation, actor_checkpoint)
                action = np.array(action, dtype=float)
                if show:
                    with prediction_container:
                        if not annotate or len(anns['seen_targets']) == 0:
                            # if not annotate:
                            selected_options, action_anns = display_action(
                                args, options, action, action_stddev, i,
                                actor_checkpoint, parameter_votes_info)
                            anns.update(action_anns)
                            # else:
                            #     selected_options = list(range(len(options)))
            if show:
                with evidence_container:
                    if actor_checkpoint != 'Choose your own actions' and (
                            not annotate or len(anns['seen_targets']) == 0):
                        anns.update(display_action_evidence(
                            args, actor_checkpoint, i, action, parameter_votes_info,
                            parameters_info, options,
                            selected_options=selected_options))
                if annotate:
                    if len(anns['seen_targets']) == 0:
                        with part4_container:
                            anns.update(annotate_timepoint_end(i, info, anns))
                    submit_annotations = st.button('Submit Annotations', key=f'submit anns {i}')
                    if submit_annotations:
                        write_annotations(
                            split, instance_name, i, observation, info,
                            reports, anns)
                        st.success(
                            'The target diagnoses were: ' + ', '.join(
                                list(info['current_targets'])))
                    st.expander('Annotations').write(get_current_annotations())
                    action_submitted = False
                else:
                    action_submitted = st.button('Next Timestep', key=f'submit {i}')
            else:
                action_submitted = False
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
container.empty()
with container.container():
    display_state(observation, info, reward)
    st.write('Done! Environment was {}.'.format(
        'terminated' if terminated else 'truncated'))
