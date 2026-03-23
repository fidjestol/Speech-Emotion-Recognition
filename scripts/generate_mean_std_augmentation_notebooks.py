from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[1]
AUGMENTATION_DIR = REPO_ROOT / "feature_augmentation" / "mean_std_oversampling"

DATASET_KEYS = [
    "mfcc_raw",
    "prosody_energy",
    "prosody_pitch",
    "bert",
    "representations",
    "rhythm_pauses",
    "tonality",
    "ssl_hubert",
    "ssl_wav2vec",
    "combined",
]


def block(text: str) -> str:
    return dedent(text).strip("\n") + "\n"


def write_notebook(path: Path, notebook) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


def dataset_title(dataset_key: str) -> str:
    return dataset_key.replace("_", " ").title()


COMMON_IMPORTS = block(
    """
    from pathlib import Path
    import json
    import os
    import sys
    from contextlib import contextmanager

    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from IPython.display import display
    from joblib import Parallel, delayed
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from tqdm.auto import tqdm

    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    np.random.seed(42)

    print('Imports loaded.')
    """
)


COMMON_CONFIG_TEMPLATE = block(
    """
    CWD = Path.cwd().resolve()
    REPO_ROOT = CWD if (CWD / 'pyproject.toml').exists() else next(
        path for path in [CWD, *CWD.parents] if (path / 'pyproject.toml').exists()
    )
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from feature_augmentation.common import (
        DEFAULT_AUGMENTATION_METHOD,
        build_classification_report_frame,
        build_confusion_matrix_frame,
        build_stage_class_count_table,
        mean_std_oversample_training_frame,
        resolve_augmentation_artifact_dir,
        summarize_class_counts,
    )
    from feature_selection.common import (
        DEFAULT_EXCLUDED_EMOTIONS,
        DEFAULT_REQUIRE_AGREEMENT,
        METADATA_CANDIDATES,
        filter_feature_frame,
        include_xxx_from_env,
        machine_name_from_env,
        resolve_cpu_parallel_config,
        resolve_feature_source,
        resolve_random_forest_jobs,
        use_variant_artifact_dirs_from_env,
        variant_name,
    )

    DATASET_KEY = '__DATASET_KEY__'
    TARGET_COL = 'emotion'
    TEST_SIZE = 0.20
    RANDOM_STATE = 42
    TARGET_PER_CLASS = 10_000
    GROUP_SIZE = 5
    AUGMENT_RANDOM_STATE = 42
    USE_NROWS = None
    REQUIRE_AGREEMENT = DEFAULT_REQUIRE_AGREEMENT
    EXCLUDED_EMOTIONS = DEFAULT_EXCLUDED_EMOTIONS
    AUGMENTATION_METHOD = DEFAULT_AUGMENTATION_METHOD

    MACHINE_NAME = machine_name_from_env()
    INCLUDE_XXX = include_xxx_from_env(default=True)
    VARIANT_NAME = variant_name(INCLUDE_XXX)
    USE_VARIANT_ARTIFACT_DIRS = use_variant_artifact_dirs_from_env(default=True)
    MODEL_PARALLEL_JOBS, MODEL_PARALLEL_MODE = resolve_cpu_parallel_config(MACHINE_NAME)
    RANDOM_FOREST_JOBS = resolve_random_forest_jobs(MACHINE_NAME)

    source_path = resolve_feature_source(REPO_ROOT, DATASET_KEY)
    artifact_dir = resolve_augmentation_artifact_dir(
        REPO_ROOT,
        include_xxx=INCLUDE_XXX,
        use_variant_dirs=USE_VARIANT_ARTIFACT_DIRS,
    )

    print(f'Repository root: {REPO_ROOT}')
    print(f'Selected source: {source_path}')
    print('Source exists:', source_path.exists())
    print(f'Machine: {MACHINE_NAME}')
    print(f'Include xxx: {INCLUDE_XXX} ({VARIANT_NAME})')
    print(f'Artifact dir: {artifact_dir}')
    print(f'CPU model parallel jobs: {MODEL_PARALLEL_JOBS} | mode={MODEL_PARALLEL_MODE}')
    print(f'Augmentation method: {AUGMENTATION_METHOD}')
    """
)


LOAD_AND_FILTER = block(
    """
    if not source_path.exists():
        raise FileNotFoundError(f'Missing source CSV: {source_path}')

    raw_df = pd.read_csv(source_path, nrows=USE_NROWS)

    if TARGET_COL not in raw_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Available: {list(raw_df.columns[:20])}...")

    rows_loaded = len(raw_df)
    df = filter_feature_frame(
        raw_df,
        target_col=TARGET_COL,
        include_xxx=INCLUDE_XXX,
        require_agreement=REQUIRE_AGREEMENT,
        excluded_emotions=EXCLUDED_EMOTIONS,
    )

    meta_cols = [c for c in df.columns if c in METADATA_CANDIDATES]
    feature_cols = [c for c in df.columns if c not in METADATA_CANDIDATES]

    raw_class_counts = summarize_class_counts(raw_df[TARGET_COL], label_name=TARGET_COL)
    filtered_class_counts = summarize_class_counts(df[TARGET_COL], label_name=TARGET_COL)

    print(f'Rows loaded: {rows_loaded}')
    print(f'Rows after variant filtering: {len(df)}')
    print(f'Metadata columns ({len(meta_cols)}): {meta_cols}')
    print(f'Feature columns ({len(feature_cols)}).')
    print('Sample feature columns:', feature_cols[:10])
    print('\\nRaw class distribution:')
    print(raw_class_counts.to_string(index=False))
    print('\\nFiltered class distribution:')
    print(filtered_class_counts.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(raw_class_counts[TARGET_COL], raw_class_counts['count'])
    axes[0].set_title('Raw Class Distribution')
    axes[0].set_xlabel(TARGET_COL)
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(filtered_class_counts[TARGET_COL], filtered_class_counts['count'])
    axes[1].set_title(f'Filtered Class Distribution ({VARIANT_NAME})')
    axes[1].set_xlabel(TARGET_COL)
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    """
)


SPLIT_AND_COUNTS = block(
    """
    model_df = df[[TARGET_COL] + feature_cols].copy()

    rows_before = len(model_df)
    model_df = model_df.dropna(subset=[TARGET_COL]).copy()
    usable_feature_cols = [c for c in feature_cols if model_df[c].notna().all()]
    dropped_feature_cols = [c for c in feature_cols if c not in usable_feature_cols]
    model_df = model_df[[TARGET_COL] + usable_feature_cols].dropna(axis=0, how='any').copy()
    rows_after = len(model_df)

    X = model_df[usable_feature_cols].astype(float)
    y = model_df[TARGET_COL].astype(str)
    all_labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    base_stage_class_counts_df = build_stage_class_count_table(
        {
            'raw_loaded': raw_df[TARGET_COL],
            'filtered': df[TARGET_COL],
            'model_ready': model_df[TARGET_COL],
            'train_reference': y_train,
            'test_reference': y_test,
        },
        label_name=TARGET_COL,
    )

    print(f'Rows before cleaning: {rows_before}')
    print(f'Rows after cleaning:  {rows_after}')
    print(f'Dropped rows:         {rows_before - rows_after}')
    print(f'Feature columns used: {len(usable_feature_cols)}')
    print(f'Dropped feature columns with NaNs: {len(dropped_feature_cols)}')
    if dropped_feature_cols:
        print('Sample dropped feature columns:', dropped_feature_cols[:10])
    print(f'Train shape: {X_train.shape}')
    print(f'Test shape:  {X_test.shape}')
    print(f'Classes: {all_labels}')

    display(base_stage_class_counts_df)

    plot_stage_order = ['filtered', 'train_reference', 'test_reference']
    plot_df = base_stage_class_counts_df[base_stage_class_counts_df['stage'].isin(plot_stage_order)].copy()
    plot_pivot = (
        plot_df
        .pivot(index=TARGET_COL, columns='stage', values='class_count')
        .fillna(0)
        .reindex(columns=plot_stage_order)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pivot.plot(kind='bar', ax=ax)
    ax.set_title('Reference Class Counts Used for Training and Evaluation')
    ax.set_xlabel(TARGET_COL)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    """
)


MODEL_AND_HELPERS = block(
    """
    MODEL_SPECS = {
        'logreg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=4000, class_weight='balanced', random_state=RANDOM_STATE)),
        ]),
        'linear_svc': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=8000)),
        ]),
        'random_forest': Pipeline([
            ('scaler', 'passthrough'),
            ('clf', RandomForestClassifier(
                n_estimators=500,
                class_weight='balanced_subsample',
                random_state=RANDOM_STATE,
                n_jobs=RANDOM_FOREST_JOBS,
            )),
        ]),
    }


    @contextmanager
    def tqdm_joblib(tqdm_object):
        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()


    def _resolve_model_jobs(requested_jobs: int, n_models: int) -> int:
        if requested_jobs == -1:
            return max(1, min(os.cpu_count() or 1, n_models))
        return max(1, min(int(requested_jobs), n_models))


    def _has_inner_parallelism(estimator) -> bool:
        params = estimator.get_params(deep=True)
        if 'clf__n_jobs' not in params:
            return False
        return params['clf__n_jobs'] not in (None, 1)


    def _fit_predict_single_model(
        model_name: str,
        estimator,
        X_train_eval,
        y_train_eval,
        X_test_eval,
        y_test_eval,
        stage_name: str,
        force_single_core_inner_model: bool,
    ):
        model = clone(estimator)

        if force_single_core_inner_model and 'clf__n_jobs' in model.get_params(deep=True):
            model.set_params(clf__n_jobs=1)

        model.fit(X_train_eval, y_train_eval)
        y_pred = model.predict(X_test_eval)

        row = {
            'stage': stage_name,
            'model': model_name,
            'accuracy': accuracy_score(y_test_eval, y_pred),
            'precision_weighted': precision_score(y_test_eval, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test_eval, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test_eval, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test_eval, y_pred, average='macro'),
        }
        return model_name, row, pd.Series(y_pred, index=y_test_eval.index)


    def _evaluate_items_sequential(
        model_items,
        X_train_eval,
        y_train_eval,
        X_test_eval,
        y_test_eval,
        stage_name: str,
        desc: str,
        force_single_core_inner_model: bool,
    ):
        return [
            _fit_predict_single_model(
                model_name,
                estimator,
                X_train_eval,
                y_train_eval,
                X_test_eval,
                y_test_eval,
                stage_name,
                force_single_core_inner_model,
            )
            for model_name, estimator in tqdm(
                model_items,
                total=len(model_items),
                desc=desc,
                leave=False,
            )
        ]


    def _evaluate_items_parallel(
        model_items,
        requested_jobs: int,
        X_train_eval,
        y_train_eval,
        X_test_eval,
        y_test_eval,
        stage_name: str,
        desc: str,
        force_single_core_inner_model: bool,
    ):
        if not model_items:
            return []

        n_jobs = _resolve_model_jobs(requested_jobs, len(model_items))
        if n_jobs == 1:
            return _evaluate_items_sequential(
                model_items,
                X_train_eval,
                y_train_eval,
                X_test_eval,
                y_test_eval,
                stage_name,
                desc,
                force_single_core_inner_model,
            )

        with tqdm_joblib(tqdm(total=len(model_items), desc=desc, leave=False)):
            return Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(_fit_predict_single_model)(
                    model_name,
                    estimator,
                    X_train_eval,
                    y_train_eval,
                    X_test_eval,
                    y_test_eval,
                    stage_name,
                    force_single_core_inner_model,
                )
                for model_name, estimator in model_items
            )


    def evaluate_models(X_train_eval, y_train_eval, X_test_eval, y_test_eval, stage_name: str):
        model_items = list(MODEL_SPECS.items())
        mode = str(MODEL_PARALLEL_MODE).strip().lower()
        if mode not in {'auto', 'outer', 'sequential'}:
            raise ValueError('MODEL_PARALLEL_MODE must be one of: auto, outer, sequential')

        requested_jobs = _resolve_model_jobs(MODEL_PARALLEL_JOBS, len(model_items))

        if mode == 'sequential' or requested_jobs == 1:
            evaluated = _evaluate_items_sequential(
                model_items,
                X_train_eval,
                y_train_eval,
                X_test_eval,
                y_test_eval,
                stage_name,
                desc=f'{stage_name}: fit/predict',
                force_single_core_inner_model=False,
            )
        elif mode == 'outer':
            evaluated = _evaluate_items_parallel(
                model_items,
                requested_jobs,
                X_train_eval,
                y_train_eval,
                X_test_eval,
                y_test_eval,
                stage_name,
                desc=f'{stage_name}: fit/predict',
                force_single_core_inner_model=True,
            )
        else:
            has_inner_parallel = any(_has_inner_parallelism(estimator) for _, estimator in model_items)
            evaluated = _evaluate_items_parallel(
                model_items,
                requested_jobs,
                X_train_eval,
                y_train_eval,
                X_test_eval,
                y_test_eval,
                stage_name,
                desc=f'{stage_name}: fit/predict',
                force_single_core_inner_model=has_inner_parallel,
            )

        rows = [row for _, row, _ in evaluated]
        preds = {model_name: pred_series for model_name, _, pred_series in evaluated}
        results_df = pd.DataFrame(rows).sort_values('f1_weighted', ascending=False, ignore_index=True)
        return results_df, preds


    def plot_confusion_heatmaps(y_true, y_pred, title_prefix: str):
        labels = list(all_labels)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[0],
        )
        axes[0].set_title(f'{title_prefix} | Counts')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[1],
            vmin=0.0,
            vmax=1.0,
        )
        axes[1].set_title(f'{title_prefix} | Row-normalized')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')

        plt.tight_layout()
        plt.show()
    """
)


BASELINE_EVAL = block(
    """
    baseline_results, baseline_preds = evaluate_models(
        X_train,
        y_train,
        X_test,
        y_test,
        stage_name='baseline',
    )

    print('Baseline metrics (all models):')
    print(baseline_results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    baseline_report_frames = []
    baseline_confusion_frames = []
    for model_name in tqdm(
        baseline_results['model'],
        total=len(baseline_results),
        desc='baseline: reports/confusion',
        leave=False,
    ):
        report_dict = classification_report(
            y_test,
            baseline_preds[model_name],
            labels=all_labels,
            target_names=all_labels,
            zero_division=0,
            output_dict=True,
        )
        print(f'\\n[{model_name}] classification report (baseline):')
        print(classification_report(
            y_test,
            baseline_preds[model_name],
            labels=all_labels,
            target_names=all_labels,
            zero_division=0,
        ))
        baseline_report_frames.append(
            build_classification_report_frame(report_dict, stage_name='baseline', model_name=model_name)
        )
        baseline_confusion_frames.append(
            build_confusion_matrix_frame(
                confusion_matrix(y_test, baseline_preds[model_name], labels=all_labels),
                labels=all_labels,
                stage_name='baseline',
                model_name=model_name,
            )
        )

    baseline_report_df = pd.concat(baseline_report_frames, ignore_index=True)
    baseline_confusion_df = pd.concat(baseline_confusion_frames, ignore_index=True)

    best_baseline_model = baseline_results.loc[0, 'model']
    plot_confusion_heatmaps(y_test, baseline_preds[best_baseline_model], f'Baseline Confusion ({best_baseline_model})')

    base_acc = float(baseline_results.loc[0, 'accuracy'])
    base_f1 = float(baseline_results.loc[0, 'f1_weighted'])
    base_f1_macro = float(baseline_results.loc[0, 'f1_macro'])
    """
)


AUGMENTATION_AND_PLOTS = block(
    """
    X_train_aug, y_train_aug, synthetic_meta_df, augmentation_summary_df = mean_std_oversample_training_frame(
        X_train,
        y_train,
        target_per_class=TARGET_PER_CLASS,
        group_size=GROUP_SIZE,
        random_state=AUGMENT_RANDOM_STATE,
        sample_with_replacement=True,
    )

    train_counts_before = summarize_class_counts(y_train, label_name=TARGET_COL)
    train_counts_after = summarize_class_counts(y_train_aug, label_name=TARGET_COL)
    synthetic_only_labels = synthetic_meta_df[TARGET_COL] if TARGET_COL in synthetic_meta_df.columns else pd.Series(dtype=str)
    synthetic_only_counts = summarize_class_counts(synthetic_only_labels, label_name=TARGET_COL)

    stage_class_counts_df = build_stage_class_count_table(
        {
            'raw_loaded': raw_df[TARGET_COL],
            'filtered': df[TARGET_COL],
            'model_ready': model_df[TARGET_COL],
            'train_reference': y_train,
            'test_reference': y_test,
            'synthetic_only': synthetic_meta_df[TARGET_COL],
            'train_augmented': y_train_aug,
        },
        label_name=TARGET_COL,
    )

    counts_compare_df = train_counts_before.merge(
        train_counts_after,
        on=TARGET_COL,
        how='outer',
        suffixes=('_before', '_after'),
    ).fillna(0)
    counts_compare_df[['count_before', 'count_after']] = counts_compare_df[['count_before', 'count_after']].astype(int)
    counts_compare_df['synthetic_added'] = counts_compare_df['count_after'] - counts_compare_df['count_before']

    if synthetic_meta_df.empty:
        synthetic_kind_counts_df = pd.DataFrame(columns=[TARGET_COL, 'synthetic_kind', 'count'])
    else:
        synthetic_kind_counts_df = (
            synthetic_meta_df
            .groupby([TARGET_COL, 'synthetic_kind'], as_index=False)
            .size()
            .rename(columns={'size': 'count'})
        )

    print(f'Original train shape:  {X_train.shape}')
    print(f'Augmented train shape: {X_train_aug.shape}')
    print(f'Synthetic rows added:  {len(synthetic_meta_df)}')
    print('Augmentation summary by class:')
    display(augmentation_summary_df)
    print('Stage-by-stage class counts:')
    display(stage_class_counts_df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].bar(counts_compare_df[TARGET_COL], counts_compare_df['count_before'])
    axes[0, 0].set_title('Training Class Distribution Before Augmentation')
    axes[0, 0].set_xlabel(TARGET_COL)
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)

    axes[0, 1].bar(counts_compare_df[TARGET_COL], counts_compare_df['count_after'])
    axes[0, 1].set_title('Training Class Distribution After Augmentation')
    axes[0, 1].set_xlabel(TARGET_COL)
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)

    axes[1, 0].bar(counts_compare_df[TARGET_COL], counts_compare_df['synthetic_added'])
    axes[1, 0].set_title('Synthetic Rows Added Per Emotion')
    axes[1, 0].set_xlabel(TARGET_COL)
    axes[1, 0].set_ylabel('Synthetic rows added')
    axes[1, 0].tick_params(axis='x', rotation=45)

    if synthetic_kind_counts_df.empty:
        axes[1, 1].text(0.5, 0.5, 'No synthetic rows generated', ha='center', va='center')
        axes[1, 1].set_axis_off()
    else:
        kind_plot = synthetic_kind_counts_df.pivot(index=TARGET_COL, columns='synthetic_kind', values='count').fillna(0)
        kind_plot.plot(kind='bar', stacked=True, ax=axes[1, 1])
        axes[1, 1].set_title('Synthetic Kind Counts by Emotion')
        axes[1, 1].set_xlabel(TARGET_COL)
        axes[1, 1].set_ylabel('Synthetic rows')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(title='Synthetic kind', loc='best')

    plt.tight_layout()
    plt.show()

    stage_plot_order = ['train_reference', 'test_reference', 'synthetic_only', 'train_augmented']
    stage_plot_df = stage_class_counts_df[stage_class_counts_df['stage'].isin(stage_plot_order)].copy()
    stage_plot_pivot = (
        stage_plot_df
        .pivot(index=TARGET_COL, columns='stage', values='class_count')
        .fillna(0)
        .reindex(columns=stage_plot_order)
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    stage_plot_pivot.plot(kind='bar', ax=ax)
    ax.set_title('Class Counts Across Reference, Synthetic, and Augmented Sets')
    ax.set_xlabel(TARGET_COL)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    """
)


AUGMENTED_EVAL = block(
    """
    augmented_results, augmented_preds = evaluate_models(
        X_train_aug,
        y_train_aug,
        X_test,
        y_test,
        stage_name='augmented',
    )

    print('Post-augmentation metrics (all models):')
    print(augmented_results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    augmented_report_frames = []
    augmented_confusion_frames = []
    for model_name in tqdm(
        augmented_results['model'],
        total=len(augmented_results),
        desc='augmented: reports/confusion',
        leave=False,
    ):
        report_dict = classification_report(
            y_test,
            augmented_preds[model_name],
            labels=all_labels,
            target_names=all_labels,
            zero_division=0,
            output_dict=True,
        )
        print(f'\\n[{model_name}] classification report (post-augmentation):')
        print(classification_report(
            y_test,
            augmented_preds[model_name],
            labels=all_labels,
            target_names=all_labels,
            zero_division=0,
        ))
        augmented_report_frames.append(
            build_classification_report_frame(report_dict, stage_name='augmented', model_name=model_name)
        )
        augmented_confusion_frames.append(
            build_confusion_matrix_frame(
                confusion_matrix(y_test, augmented_preds[model_name], labels=all_labels),
                labels=all_labels,
                stage_name='augmented',
                model_name=model_name,
            )
        )

    augmented_report_df = pd.concat(augmented_report_frames, ignore_index=True)
    augmented_confusion_df = pd.concat(augmented_confusion_frames, ignore_index=True)

    best_augmented_model = augmented_results.loc[0, 'model']
    plot_confusion_heatmaps(y_test, augmented_preds[best_augmented_model], f'Augmented Confusion ({best_augmented_model})')

    comparison_df = baseline_results[['model', 'accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'f1_macro']].merge(
        augmented_results[['model', 'accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'f1_macro']],
        on='model',
        suffixes=('_baseline', '_augmented'),
    )
    comparison_df['accuracy_delta'] = comparison_df['accuracy_augmented'] - comparison_df['accuracy_baseline']
    comparison_df['precision_weighted_delta'] = comparison_df['precision_weighted_augmented'] - comparison_df['precision_weighted_baseline']
    comparison_df['recall_weighted_delta'] = comparison_df['recall_weighted_augmented'] - comparison_df['recall_weighted_baseline']
    comparison_df['f1_weighted_delta'] = comparison_df['f1_weighted_augmented'] - comparison_df['f1_weighted_baseline']
    comparison_df['f1_macro_delta'] = comparison_df['f1_macro_augmented'] - comparison_df['f1_macro_baseline']
    comparison_df = comparison_df.sort_values('f1_weighted_augmented', ascending=False, ignore_index=True)

    print('\\nMetric delta by model (augmentation - baseline):')
    print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    aug_acc = float(augmented_results.loc[0, 'accuracy'])
    aug_f1 = float(augmented_results.loc[0, 'f1_weighted'])
    aug_f1_macro = float(augmented_results.loc[0, 'f1_macro'])

    fig, ax = plt.subplots(figsize=(10, 5))
    metric_delta_plot = comparison_df.set_index('model')[
        ['accuracy_delta', 'precision_weighted_delta', 'recall_weighted_delta', 'f1_weighted_delta', 'f1_macro_delta']
    ]
    metric_delta_plot.plot(kind='bar', ax=ax)
    ax.axhline(0.0, color='black', linewidth=1)
    ax.set_title('Metric Delta by Model (Augmented - Baseline)')
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric delta')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.show()

    best_models = [best_baseline_model, best_augmented_model]
    best_report_compare_df = (
        pd.concat([baseline_report_df, augmented_report_df], ignore_index=True)
        .query("label_type == 'class' and model in @best_models")
        .copy()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    precision_plot_df = best_report_compare_df.pivot_table(
        index='label',
        columns='stage',
        values='precision',
        aggfunc='first',
    ).fillna(0.0)
    precision_plot_df.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Per-class Precision: Best Baseline vs Best Augmented')
    axes[0].set_xlabel(TARGET_COL)
    axes[0].set_ylabel('Precision')
    axes[0].tick_params(axis='x', rotation=45)

    f1_plot_df = best_report_compare_df.pivot_table(
        index='label',
        columns='stage',
        values='f1_score',
        aggfunc='first',
    ).fillna(0.0)
    f1_plot_df.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Per-class F1: Best Baseline vs Best Augmented')
    axes[1].set_xlabel(TARGET_COL)
    axes[1].set_ylabel('F1 score')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    """
)


SAVE_ARTIFACTS = block(
    """
    out_dir = artifact_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_aug_path = out_dir / f'{DATASET_KEY}_X_train_augmented.csv'
    train_aug_labels_path = out_dir / f'{DATASET_KEY}_y_train_augmented.csv'
    test_ref_path = out_dir / f'{DATASET_KEY}_X_test_reference.csv'
    test_ref_labels_path = out_dir / f'{DATASET_KEY}_y_test_reference.csv'
    synthetic_meta_path = out_dir / f'{DATASET_KEY}_synthetic_metadata.csv'
    summary_path = out_dir / f'{DATASET_KEY}_augmentation_summary.csv'
    class_counts_path = out_dir / f'{DATASET_KEY}_class_counts.csv'
    synthetic_kind_counts_path = out_dir / f'{DATASET_KEY}_synthetic_kind_counts.csv'
    baseline_metrics_path = out_dir / f'{DATASET_KEY}_baseline_model_metrics.csv'
    augmented_metrics_path = out_dir / f'{DATASET_KEY}_augmented_model_metrics.csv'
    comparison_path = out_dir / f'{DATASET_KEY}_model_metric_deltas.csv'
    classification_reports_path = out_dir / f'{DATASET_KEY}_classification_reports.csv'
    confusion_matrices_path = out_dir / f'{DATASET_KEY}_confusion_matrices.csv'
    prediction_path = out_dir / f'{DATASET_KEY}_test_predictions.csv'
    meta_path = out_dir / f'{DATASET_KEY}_run_metadata.json'

    X_train_aug.to_csv(train_aug_path, index=False)
    y_train_aug.to_frame(name=TARGET_COL).to_csv(train_aug_labels_path, index=False)
    X_test.to_csv(test_ref_path, index=False)
    y_test.to_frame(name=TARGET_COL).to_csv(test_ref_labels_path, index=False)
    synthetic_meta_df.to_csv(synthetic_meta_path, index=False)
    augmentation_summary_df.to_csv(summary_path, index=False)
    stage_class_counts_df.to_csv(class_counts_path, index=False)
    synthetic_kind_counts_df.to_csv(synthetic_kind_counts_path, index=False)
    baseline_results.to_csv(baseline_metrics_path, index=False)
    augmented_results.to_csv(augmented_metrics_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)

    classification_reports_df = pd.concat([baseline_report_df, augmented_report_df], ignore_index=True)
    classification_reports_df.to_csv(classification_reports_path, index=False)

    confusion_matrices_df = pd.concat([baseline_confusion_df, augmented_confusion_df], ignore_index=True)
    confusion_matrices_df.to_csv(confusion_matrices_path, index=False)

    prediction_df = pd.DataFrame({
        'test_index': X_test.index.astype(int),
        TARGET_COL: y_test.reset_index(drop=True),
    })
    for model_name in baseline_results['model']:
        prediction_df[f'baseline__{model_name}'] = baseline_preds[model_name].reset_index(drop=True)
    for model_name in augmented_results['model']:
        prediction_df[f'augmented__{model_name}'] = augmented_preds[model_name].reset_index(drop=True)
    prediction_df.to_csv(prediction_path, index=False)

    baseline_records = baseline_results.to_dict(orient='records')
    augmented_records = augmented_results.to_dict(orient='records')
    comparison_records = comparison_df.to_dict(orient='records')
    for collection in (baseline_records, augmented_records, comparison_records):
        for row in collection:
            for key, value in list(row.items()):
                if isinstance(value, (np.floating, float)):
                    row[key] = float(value)

    run_meta = {
        'dataset_key': DATASET_KEY,
        'source_path': str(source_path),
        'artifact_dir': str(out_dir),
        'target_col': TARGET_COL,
        'machine_name': MACHINE_NAME,
        'include_xxx': bool(INCLUDE_XXX),
        'variant_name': VARIANT_NAME,
        'require_agreement': bool(REQUIRE_AGREEMENT),
        'excluded_emotions': list(EXCLUDED_EMOTIONS),
        'use_variant_artifact_dirs': bool(USE_VARIANT_ARTIFACT_DIRS),
        'rows_loaded': int(rows_loaded),
        'rows_after_filter': int(len(df)),
        'rows_used': int(len(model_df)),
        'train_rows_reference': int(len(X_train)),
        'test_rows_reference': int(len(X_test)),
        'original_features': int(X_train.shape[1]),
        'feature_columns_used': int(len(usable_feature_cols)),
        'dropped_feature_columns': int(len(dropped_feature_cols)),
        'classes': list(all_labels),
        'random_state': RANDOM_STATE,
        'test_size': TEST_SIZE,
        'model_parallel_jobs': int(MODEL_PARALLEL_JOBS),
        'model_parallel_mode': MODEL_PARALLEL_MODE,
        'augmentation_method': AUGMENTATION_METHOD,
        'augmentation_target_per_class': int(TARGET_PER_CLASS),
        'augmentation_group_size': int(GROUP_SIZE),
        'augmentation_random_state': int(AUGMENT_RANDOM_STATE),
        'synthetic_rows_added': int(len(synthetic_meta_df)),
        'train_rows_before_augmentation': int(len(X_train)),
        'train_rows_after_augmentation': int(len(X_train_aug)),
        'best_baseline_model': str(best_baseline_model),
        'best_augmented_model': str(best_augmented_model),
        'baseline_accuracy': float(base_acc),
        'baseline_f1_weighted': float(base_f1),
        'baseline_f1_macro': float(base_f1_macro),
        'augmented_accuracy': float(aug_acc),
        'augmented_f1_weighted': float(aug_f1),
        'augmented_f1_macro': float(aug_f1_macro),
        'baseline_model_metrics': baseline_records,
        'augmented_model_metrics': augmented_records,
        'model_metric_deltas': comparison_records,
        'artifact_files': {
            'train_augmented': str(train_aug_path),
            'train_augmented_labels': str(train_aug_labels_path),
            'test_reference': str(test_ref_path),
            'test_reference_labels': str(test_ref_labels_path),
            'synthetic_metadata': str(synthetic_meta_path),
            'augmentation_summary': str(summary_path),
            'class_counts': str(class_counts_path),
            'synthetic_kind_counts': str(synthetic_kind_counts_path),
            'baseline_model_metrics': str(baseline_metrics_path),
            'augmented_model_metrics': str(augmented_metrics_path),
            'model_metric_deltas': str(comparison_path),
            'classification_reports': str(classification_reports_path),
            'confusion_matrices': str(confusion_matrices_path),
            'test_predictions': str(prediction_path),
        },
    }

    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(run_meta, f, indent=2)

    print('Saved artifacts:')
    print('-', train_aug_path)
    print('-', train_aug_labels_path)
    print('-', test_ref_path)
    print('-', test_ref_labels_path)
    print('-', synthetic_meta_path)
    print('-', summary_path)
    print('-', class_counts_path)
    print('-', synthetic_kind_counts_path)
    print('-', baseline_metrics_path)
    print('-', augmented_metrics_path)
    print('-', comparison_path)
    print('-', classification_reports_path)
    print('-', confusion_matrices_path)
    print('-', prediction_path)
    print('-', meta_path)
    """
)


DELTA_SUMMARY = block(
    """
    delta_cols = [
        'model',
        'accuracy_baseline',
        'accuracy_augmented',
        'accuracy_delta',
        'precision_weighted_baseline',
        'precision_weighted_augmented',
        'precision_weighted_delta',
        'recall_weighted_baseline',
        'recall_weighted_augmented',
        'recall_weighted_delta',
        'f1_weighted_baseline',
        'f1_weighted_augmented',
        'f1_weighted_delta',
        'f1_macro_baseline',
        'f1_macro_augmented',
        'f1_macro_delta',
    ]

    delta_table = comparison_df[delta_cols].copy().sort_values('f1_weighted_delta', ascending=False, ignore_index=True)

    print('Delta table (augmentation - baseline):')
    print(delta_table.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    best_gain_row = delta_table.iloc[0]
    print(
        f"\\nBest weighted-F1 delta after augmentation: {best_gain_row['model']} | "
        f"Delta F1-weighted={best_gain_row['f1_weighted_delta']:+.4f}, "
        f"Delta Macro-F1={best_gain_row['f1_macro_delta']:+.4f}, "
        f"Delta Accuracy={best_gain_row['accuracy_delta']:+.4f}"
    )
    """
)


RUNNER_NOTEBOOK_CODE = block(
    """
    from pathlib import Path
    import json
    import os
    import re
    import sys
    import time
    from contextlib import contextmanager

    import nbformat
    import pandas as pd
    from nbclient import NotebookClient
    from tqdm.auto import tqdm

    CWD = Path.cwd().resolve()
    REPO_ROOT = CWD if (CWD / 'pyproject.toml').exists() else next(
        path for path in [CWD, *CWD.parents] if (path / 'pyproject.toml').exists()
    )
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from feature_augmentation.common import resolve_augmentation_artifact_dir
    from feature_selection.common import machine_name_from_env, resolve_feature_source, variant_name, variants_to_run


    def find_repo_root(start: Path) -> Path:
        for path in [start, *start.parents]:
            if (path / 'pyproject.toml').exists():
                return path
        raise FileNotFoundError('Could not locate repository root (missing pyproject.toml).')


    @contextmanager
    def pushd(target_dir: Path):
        prev = Path.cwd()
        os.chdir(target_dir)
        try:
            yield
        finally:
            os.chdir(prev)


    @contextmanager
    def temp_environ(updates: dict[str, str]):
        previous = {key: os.environ.get(key) for key in updates}
        for key, value in updates.items():
            os.environ[key] = str(value)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


    REPO_ROOT = find_repo_root(Path.cwd().resolve())
    NOTEBOOK_DIR = REPO_ROOT / 'feature_augmentation' / 'mean_std_oversampling'
    RUNNER_NAME = 'run_all_augmentation.ipynb'
    EXECUTE_TIMEOUT = None
    KERNEL_NAME = 'python3'
    CONTINUE_ON_ERROR = True
    SKIP_EXISTING = True
    USE_VARIANT_ARTIFACT_DIRS = True
    MACHINE_NAME = machine_name_from_env()
    VARIANTS = variants_to_run(default_both=True)
    SKIP_NOTEBOOKS = {'visualize_augmentation_results.ipynb'}

    targets = sorted(
        p for p in NOTEBOOK_DIR.glob('augment_*.ipynb')
        if p.name not in {RUNNER_NAME, *SKIP_NOTEBOOKS}
    )

    print(f'Notebook dir: {NOTEBOOK_DIR}')
    print(f'Runner: {RUNNER_NAME}')
    print(f'Machine: {MACHINE_NAME}')
    print('Variants:', [variant_name(include_xxx) for include_xxx in VARIANTS])
    print('Skipped notebooks:', sorted(SKIP_NOTEBOOKS))
    print(f'Targets: {len(targets)}')
    for p in targets:
        print(f' - {p.name}')

    DATASET_KEY_PATTERN = re.compile(r"^DATASET_KEY\\s*=\\s*['\\"]([^'\\"]+)['\\"]\\s*$", re.MULTILINE)


    def extract_dataset_key(notebook_path: Path) -> str | None:
        notebook = nbformat.read(notebook_path, as_version=4)
        for cell in notebook.cells:
            if cell.get('cell_type') != 'code':
                continue
            match = DATASET_KEY_PATTERN.search(cell.get('source', ''))
            if match:
                return match.group(1)
        return None


    NOTEBOOK_DATASET_KEYS = {path: extract_dataset_key(path) for path in targets}

    MISSING_SOURCE_NOTEBOOKS = {
        path.name
        for path, dataset_key in NOTEBOOK_DATASET_KEYS.items()
        if dataset_key and not resolve_feature_source(REPO_ROOT, dataset_key).exists()
    }
    if MISSING_SOURCE_NOTEBOOKS:
        print('Skipping notebooks with missing source CSVs:', sorted(MISSING_SOURCE_NOTEBOOKS))
        targets = [path for path in targets if path.name not in MISSING_SOURCE_NOTEBOOKS]
        NOTEBOOK_DATASET_KEYS = {path: NOTEBOOK_DATASET_KEYS[path] for path in targets}


    def resolve_existing_meta_path(notebook_path: Path, *, include_xxx: bool) -> Path | None:
        dataset_key = NOTEBOOK_DATASET_KEYS.get(notebook_path)
        if not dataset_key:
            return None
        artifact_dir = resolve_augmentation_artifact_dir(
            REPO_ROOT,
            include_xxx=include_xxx,
            use_variant_dirs=USE_VARIANT_ARTIFACT_DIRS,
        )
        return artifact_dir / f'{dataset_key}_run_metadata.json'


    print(f'Skip existing artifact runs: {SKIP_EXISTING}')
    """
)


RUNNER_EXECUTION_CODE = block(
    """
    results: list[dict[str, object]] = []

    for include_xxx in VARIANTS:
        current_variant = variant_name(include_xxx)

        for notebook_path in tqdm(
            targets,
            total=len(targets),
            desc=f'Running notebooks | {current_variant}',
        ):
            row = {
                'variant': current_variant,
                'notebook': notebook_path.name,
                'dataset_key': NOTEBOOK_DATASET_KEYS.get(notebook_path),
                'status': 'pending',
                'seconds': None,
                'error': None,
            }
            started = time.perf_counter()
            existing_meta_path = resolve_existing_meta_path(notebook_path, include_xxx=include_xxx)
            row['meta_path'] = str(existing_meta_path) if existing_meta_path is not None else None

            if SKIP_EXISTING and existing_meta_path is not None and existing_meta_path.exists():
                row['status'] = 'skipped_existing'
                row['seconds'] = 0.0
                results.append(row)
                continue

            try:
                nb = nbformat.read(notebook_path, as_version=4)
                with pushd(NOTEBOOK_DIR), temp_environ({
                    'SER_MACHINE': MACHINE_NAME,
                    'SER_INCLUDE_XXX': 'true' if include_xxx else 'false',
                    'SER_RUN_BOTH_XXX_VARIANTS': '0',
                    'SER_USE_VARIANT_ARTIFACT_DIRS': '1',
                }):
                    client = NotebookClient(
                        nb,
                        timeout=EXECUTE_TIMEOUT,
                        kernel_name=KERNEL_NAME,
                    )
                    client.execute()
                row['status'] = 'ok'
            except Exception as exc:  # noqa: BLE001
                row['status'] = 'failed'
                row['error'] = f'{type(exc).__name__}: {exc}'
                if not CONTINUE_ON_ERROR:
                    row['seconds'] = round(time.perf_counter() - started, 3)
                    results.append(row)
                    raise
            finally:
                if row['seconds'] is None:
                    row['seconds'] = round(time.perf_counter() - started, 3)
                results.append(row)

    results_df = pd.DataFrame(results)
    display(results_df)

    if not results_df.empty:
        print('\\nRun status counts:')
        print(results_df['status'].value_counts())
    """
)


VIS_NOTEBOOK_IMPORTS = block(
    """
    from functools import lru_cache
    from pathlib import Path
    import json
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from IPython.display import display

    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')

    CWD = Path.cwd().resolve()
    REPO_ROOT = CWD if (CWD / 'pyproject.toml').exists() else next(
        path for path in [CWD, *CWD.parents] if (path / 'pyproject.toml').exists()
    )
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from feature_augmentation.common import resolve_augmentation_artifact_dir

    AUGMENTATION_DIR = REPO_ROOT / 'feature_augmentation' / 'mean_std_oversampling'
    DEFAULT_ARTIFACT_DIR = AUGMENTATION_DIR / 'artifacts'
    WITH_XXX_DIR = resolve_augmentation_artifact_dir(REPO_ROOT, include_xxx=True, use_variant_dirs=True)
    WITHOUT_XXX_DIR = resolve_augmentation_artifact_dir(REPO_ROOT, include_xxx=False, use_variant_dirs=True)

    ARTIFACT_GROUPS = {}
    if WITH_XXX_DIR.exists():
        ARTIFACT_GROUPS['with_xxx'] = WITH_XXX_DIR
    if WITHOUT_XXX_DIR.exists():
        ARTIFACT_GROUPS['without_xxx'] = WITHOUT_XXX_DIR
    if not ARTIFACT_GROUPS and DEFAULT_ARTIFACT_DIR.exists():
        ARTIFACT_GROUPS['current'] = DEFAULT_ARTIFACT_DIR

    FOCUS_DATASET = None
    FOCUS_GROUP = None
    SAVE_COMPARISON_CSV = True

    for group_name, artifact_dir in ARTIFACT_GROUPS.items():
        print(f'{group_name}: {artifact_dir} | exists={artifact_dir.exists()}')
    """
)


VIS_HELPERS = block(
    """
    @lru_cache(maxsize=None)
    def _load_json(path_str: str) -> dict:
        return json.loads(Path(path_str).read_text(encoding='utf-8'))


    def read_csv_if_exists(path_str: str | None) -> pd.DataFrame:
        if not path_str:
            return pd.DataFrame()
        path = Path(path_str)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)


    def best_metric_row(meta: dict, stage: str) -> dict:
        if stage not in {'baseline', 'augmented'}:
            raise ValueError(f'Unsupported stage: {stage}')

        model_key = 'best_baseline_model' if stage == 'baseline' else 'best_augmented_model'
        rows_key = 'baseline_model_metrics' if stage == 'baseline' else 'augmented_model_metrics'
        best_model = meta.get(model_key)

        for row in meta.get(rows_key, []):
            if row.get('model') == best_model:
                return row

        rows = meta.get(rows_key, [])
        if rows:
            return max(rows, key=lambda row: row.get('f1_macro', float('-inf')))
        return {}


    def support_columns(meta: dict) -> dict[str, int]:
        artifact_files = meta.get('artifact_files', {})
        class_counts_df = read_csv_if_exists(artifact_files.get('class_counts'))
        if class_counts_df.empty:
            return {}

        rows = {}
        for stage_name in ['train_reference', 'test_reference', 'synthetic_only', 'train_augmented']:
            stage_df = class_counts_df[class_counts_df['stage'] == stage_name]
            for _, row in stage_df.iterrows():
                label = str(row[meta.get('target_col', 'emotion')])
                rows[f'{stage_name}_support_{label}'] = int(row['class_count'])
        return rows


    def load_group_summary(group_name: str, artifact_dir: Path) -> pd.DataFrame:
        rows = []
        for meta_path in sorted(artifact_dir.glob('*_run_metadata.json')):
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            baseline_best = best_metric_row(meta, 'baseline')
            augmented_best = best_metric_row(meta, 'augmented')

            row = {
                'group': group_name,
                'dataset_key': meta['dataset_key'],
                'rows_loaded': meta.get('rows_loaded'),
                'rows_after_filter': meta.get('rows_after_filter'),
                'rows_used': meta.get('rows_used'),
                'train_rows_reference': meta.get('train_rows_reference', meta.get('train_rows_before_augmentation')),
                'test_rows_reference': meta.get('test_rows_reference'),
                'train_rows_after_augmentation': meta.get('train_rows_after_augmentation'),
                'synthetic_rows_added': meta.get('synthetic_rows_added'),
                'feature_columns_used': meta.get('feature_columns_used'),
                'best_baseline_model': meta.get('best_baseline_model'),
                'best_augmented_model': meta.get('best_augmented_model'),
                'baseline_accuracy': baseline_best.get('accuracy', meta.get('baseline_accuracy')),
                'baseline_precision_weighted': baseline_best.get('precision_weighted'),
                'baseline_recall_weighted': baseline_best.get('recall_weighted'),
                'baseline_f1_weighted': baseline_best.get('f1_weighted', meta.get('baseline_f1_weighted')),
                'baseline_f1_macro': baseline_best.get('f1_macro', meta.get('baseline_f1_macro')),
                'augmented_accuracy': augmented_best.get('accuracy', meta.get('augmented_accuracy')),
                'augmented_precision_weighted': augmented_best.get('precision_weighted'),
                'augmented_recall_weighted': augmented_best.get('recall_weighted'),
                'augmented_f1_weighted': augmented_best.get('f1_weighted', meta.get('augmented_f1_weighted')),
                'augmented_f1_macro': augmented_best.get('f1_macro', meta.get('augmented_f1_macro')),
                'artifact_dir': meta.get('artifact_dir'),
                'meta_path': str(meta_path),
            }
            row.update(support_columns(meta))
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df['accuracy_delta'] = df['augmented_accuracy'] - df['baseline_accuracy']
        df['precision_weighted_delta'] = df['augmented_precision_weighted'] - df['baseline_precision_weighted']
        df['recall_weighted_delta'] = df['augmented_recall_weighted'] - df['baseline_recall_weighted']
        df['f1_weighted_delta'] = df['augmented_f1_weighted'] - df['baseline_f1_weighted']
        df['f1_macro_delta'] = df['augmented_f1_macro'] - df['baseline_f1_macro']
        return df.sort_values(['group', 'f1_macro_delta'], ascending=[True, False], ignore_index=True)


    def load_model_delta_table(group_name: str, artifact_dir: Path) -> pd.DataFrame:
        rows = []
        for meta_path in sorted(artifact_dir.glob('*_run_metadata.json')):
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            for row in meta.get('model_metric_deltas', []):
                current = {'group': group_name, 'dataset_key': meta['dataset_key']}
                current.update(row)
                rows.append(current)
        return pd.DataFrame(rows)


    def build_pairwise_comparison(summary_df: pd.DataFrame, left_group: str, right_group: str) -> pd.DataFrame:
        left_df = summary_df[summary_df['group'] == left_group].copy()
        right_df = summary_df[summary_df['group'] == right_group].copy()
        if left_df.empty or right_df.empty:
            return pd.DataFrame()

        merged = left_df.merge(
            right_df,
            on='dataset_key',
            suffixes=(f'|{left_group}', f'|{right_group}'),
        )
        merged[f'augmented_f1_macro_change_{right_group}_minus_{left_group}'] = (
            merged[f'augmented_f1_macro|{right_group}'] - merged[f'augmented_f1_macro|{left_group}']
        )
        merged[f'augmentation_gain_change_{right_group}_minus_{left_group}'] = (
            merged[f'f1_macro_delta|{right_group}'] - merged[f'f1_macro_delta|{left_group}']
        )
        return merged


    def load_artifact_meta(dataset_key: str, group_name: str) -> dict:
        artifact_dir = ARTIFACT_GROUPS[group_name]
        meta_path = artifact_dir / f'{dataset_key}_run_metadata.json'
        if not meta_path.exists():
            raise FileNotFoundError(f'Missing metadata for {dataset_key} | {group_name}: {meta_path}')
        return json.loads(meta_path.read_text(encoding='utf-8'))


    def load_stage_counts(dataset_key: str, group_name: str) -> pd.DataFrame:
        meta = load_artifact_meta(dataset_key, group_name)
        return read_csv_if_exists(meta.get('artifact_files', {}).get('class_counts'))


    def load_synthetic_kind_counts(dataset_key: str, group_name: str) -> pd.DataFrame:
        meta = load_artifact_meta(dataset_key, group_name)
        return read_csv_if_exists(meta.get('artifact_files', {}).get('synthetic_kind_counts'))


    def load_reports(dataset_key: str, group_name: str) -> pd.DataFrame:
        meta = load_artifact_meta(dataset_key, group_name)
        return read_csv_if_exists(meta.get('artifact_files', {}).get('classification_reports'))


    def load_confusions(dataset_key: str, group_name: str) -> pd.DataFrame:
        meta = load_artifact_meta(dataset_key, group_name)
        return read_csv_if_exists(meta.get('artifact_files', {}).get('confusion_matrices'))
    """
)


VIS_SUMMARY = block(
    """
    summary_frames = []
    delta_frames = []

    for group_name, artifact_dir in ARTIFACT_GROUPS.items():
        if artifact_dir.exists():
            summary_frames.append(load_group_summary(group_name, artifact_dir))
            delta_frames.append(load_model_delta_table(group_name, artifact_dir))

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    model_delta_df = pd.concat(delta_frames, ignore_index=True) if delta_frames else pd.DataFrame()

    if summary_df.empty:
        print('No augmentation artifact metadata found.')
    else:
        display(summary_df)
    """
)


VIS_OVERVIEW_PLOTS = block(
    """
    if not summary_df.empty:
        for group_name in summary_df['group'].unique():
            group_df = summary_df[summary_df['group'] == group_name].copy().sort_values('f1_macro_delta', ascending=True)
            fig, axes = plt.subplots(2, 2, figsize=(18, 10))

            axes[0, 0].barh(group_df['dataset_key'], group_df['f1_macro_delta'])
            axes[0, 0].axvline(0.0, color='black', linewidth=1)
            axes[0, 0].set_title(f'Macro-F1 Delta (Augmented - Baseline) | {group_name}')
            axes[0, 0].set_xlabel('Macro-F1 delta')

            axes[0, 1].barh(group_df['dataset_key'], group_df['synthetic_rows_added'])
            axes[0, 1].set_title(f'Synthetic Rows Added | {group_name}')
            axes[0, 1].set_xlabel('Synthetic rows')

            width = 0.38
            y_pos = np.arange(len(group_df))
            axes[1, 0].barh(y_pos - width / 2, group_df['baseline_f1_weighted'], height=width, label='baseline')
            axes[1, 0].barh(y_pos + width / 2, group_df['augmented_f1_weighted'], height=width, label='augmented')
            axes[1, 0].set_yticks(y_pos, group_df['dataset_key'])
            axes[1, 0].set_title(f'Best Weighted F1 Before vs After | {group_name}')
            axes[1, 0].set_xlabel('Weighted F1')
            axes[1, 0].legend()

            axes[1, 1].barh(y_pos - width / 2, group_df['train_rows_reference'], height=width, label='train reference')
            axes[1, 1].barh(y_pos + width / 2, group_df['train_rows_after_augmentation'], height=width, label='train augmented')
            axes[1, 1].set_yticks(y_pos, group_df['dataset_key'])
            axes[1, 1].set_title(f'Train Rows Before vs After Augmentation | {group_name}')
            axes[1, 1].set_xlabel('Row count')
            axes[1, 1].legend()

            plt.tight_layout()
            plt.show()
    else:
        print('No augmentation artifact metadata found.')
    """
)


VIS_MODEL_DELTA = block(
    """
    if not model_delta_df.empty:
        for metric_col in ['accuracy_delta', 'f1_weighted_delta', 'f1_macro_delta']:
            pivot_df = model_delta_df.pivot_table(
                index='dataset_key',
                columns=['group', 'model'],
                values=metric_col,
                aggfunc='first',
            ).sort_index()
            fig, ax = plt.subplots(figsize=(max(12, 1.2 * len(pivot_df.columns)), max(6, 0.45 * len(pivot_df.index))))
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='coolwarm', center=0.0, ax=ax)
            ax.set_title(f'Model-level {metric_col} Heatmap')
            plt.tight_layout()
            plt.show()
    else:
        print('No model delta table found.')
    """
)


VIS_PAIRWISE = block(
    """
    required_groups = {'with_xxx', 'without_xxx'}
    available_groups = set(summary_df['group'].unique()) if not summary_df.empty else set()

    if not required_groups.issubset(available_groups):
        print('Configured artifact groups do not yet include both `with_xxx` and `without_xxx`.')
    else:
        xxx_compare_df = build_pairwise_comparison(summary_df, 'with_xxx', 'without_xxx')
        display(xxx_compare_df)

        plot_df = xxx_compare_df.sort_values('augmented_f1_macro_change_without_xxx_minus_with_xxx', ascending=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        axes[0].barh(plot_df['dataset_key'], plot_df['augmented_f1_macro_change_without_xxx_minus_with_xxx'])
        axes[0].axvline(0.0, color='black', linewidth=1)
        axes[0].set_title('Augmented Macro-F1 Change: without_xxx - with_xxx')
        axes[0].set_xlabel('Macro-F1 change')

        axes[1].barh(plot_df['dataset_key'], plot_df['augmentation_gain_change_without_xxx_minus_with_xxx'])
        axes[1].axvline(0.0, color='black', linewidth=1)
        axes[1].set_title('Augmentation Benefit Change: without_xxx - with_xxx')
        axes[1].set_xlabel('Change in augmentation macro-F1 gain')

        plt.tight_layout()
        plt.show()

        if SAVE_COMPARISON_CSV:
            out_path = DEFAULT_ARTIFACT_DIR / 'augmentation_xxx_artifact_comparison.csv'
            xxx_compare_df.to_csv(out_path, index=False)
            print(f'Saved comparison CSV: {out_path}')
    """
)


VIS_FOCUS = block(
    """
    if summary_df.empty:
        print('No artifact data to inspect.')
    else:
        selected_dataset = FOCUS_DATASET or summary_df.iloc[0]['dataset_key']
        selected_group = FOCUS_GROUP or summary_df[summary_df['dataset_key'] == selected_dataset].iloc[0]['group']

        focus_meta = load_artifact_meta(selected_dataset, selected_group)
        focus_counts_df = load_stage_counts(selected_dataset, selected_group)
        focus_kind_df = load_synthetic_kind_counts(selected_dataset, selected_group)
        focus_report_df = load_reports(selected_dataset, selected_group)
        focus_conf_df = load_confusions(selected_dataset, selected_group)

        print(f'Focused dataset: {selected_dataset} | group={selected_group}')
        display(focus_counts_df)

        stage_order = ['raw_loaded', 'filtered', 'model_ready', 'train_reference', 'test_reference', 'synthetic_only', 'train_augmented']
        stage_plot_df = focus_counts_df[focus_counts_df['stage'].isin(stage_order)].copy()
        stage_pivot = (
            stage_plot_df
            .pivot(index=focus_meta.get('target_col', 'emotion'), columns='stage', values='class_count')
            .fillna(0)
            .reindex(columns=stage_order)
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        stage_pivot.plot(kind='bar', ax=ax)
        ax.set_title(f'Class Counts by Stage | {selected_dataset} | {selected_group}')
        ax.set_xlabel(focus_meta.get('target_col', 'emotion'))
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()

        if not focus_kind_df.empty:
            kind_pivot = focus_kind_df.pivot(index=focus_meta.get('target_col', 'emotion'), columns='synthetic_kind', values='count').fillna(0)
            fig, ax = plt.subplots(figsize=(10, 5))
            kind_pivot.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f'Synthetic Kind Mix by Emotion | {selected_dataset} | {selected_group}')
            ax.set_xlabel(focus_meta.get('target_col', 'emotion'))
            ax.set_ylabel('Synthetic rows')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()

        best_models = [
            focus_meta.get('best_baseline_model'),
            focus_meta.get('best_augmented_model'),
        ]
        report_focus_df = focus_report_df[
            (focus_report_df['label_type'] == 'class')
            & (focus_report_df['model'].isin(best_models))
        ].copy()

        if not report_focus_df.empty:
            for metric_name in ['precision', 'recall', 'f1_score', 'support']:
                plot_df = report_focus_df.pivot_table(
                    index='label',
                    columns='stage',
                    values=metric_name,
                    aggfunc='first',
                ).fillna(0.0)
                fig, ax = plt.subplots(figsize=(10, 4))
                plot_df.plot(kind='bar', ax=ax)
                ax.set_title(f'Per-class {metric_name} | {selected_dataset} | {selected_group}')
                ax.set_xlabel(focus_meta.get('target_col', 'emotion'))
                ax.set_ylabel(metric_name)
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                plt.show()

        label_order = [str(label) for label in focus_meta.get('classes', [])]
        for stage_name, model_name in [('baseline', focus_meta.get('best_baseline_model')), ('augmented', focus_meta.get('best_augmented_model'))]:
            stage_conf_df = focus_conf_df[
                (focus_conf_df['stage'] == stage_name)
                & (focus_conf_df['model'] == model_name)
            ].copy()
            if stage_conf_df.empty:
                continue
            pivot_df = stage_conf_df.pivot(index='true_label', columns='predicted_label', values='count').fillna(0)
            pivot_df = pivot_df.reindex(index=label_order, columns=label_order, fill_value=0)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap='Blues', ax=axes[0])
            axes[0].set_title(f'{stage_name.title()} Confusion Counts | {model_name}')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('True')

            norm_df = pivot_df.div(pivot_df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            sns.heatmap(norm_df, annot=True, fmt='.2f', cmap='Blues', vmin=0.0, vmax=1.0, ax=axes[1])
            axes[1].set_title(f'{stage_name.title()} Confusion Row-normalized | {model_name}')
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('True')
            plt.tight_layout()
            plt.show()
    """
)


def build_augmentation_notebook(dataset_key: str):
    title = dataset_title(dataset_key)
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            block(
                f"""
                # Mean +/- STD Oversampling: {title}

                This notebook mirrors the baseline-versus-post-change pattern used in the feature-selection notebooks, but for augmentation.

                It:
                - loads the `{dataset_key}` feature family
                - evaluates baseline models on a held-out test split
                - augments only the training split with class-conditional `mean - std`, `mean`, and `mean + std` synthetic rows
                - evaluates the augmented training set against the same untouched test set
                - saves richer artifacts for downstream visualization and audit
                """
            )
        ),
        nbf.v4.new_code_cell(COMMON_IMPORTS),
        nbf.v4.new_code_cell(COMMON_CONFIG_TEMPLATE.replace("__DATASET_KEY__", dataset_key)),
        nbf.v4.new_code_cell(LOAD_AND_FILTER),
        nbf.v4.new_code_cell(SPLIT_AND_COUNTS),
        nbf.v4.new_code_cell(MODEL_AND_HELPERS),
        nbf.v4.new_code_cell(BASELINE_EVAL),
        nbf.v4.new_code_cell(AUGMENTATION_AND_PLOTS),
        nbf.v4.new_code_cell(AUGMENTED_EVAL),
        nbf.v4.new_code_cell(SAVE_ARTIFACTS),
        nbf.v4.new_code_cell(DELTA_SUMMARY),
    ]
    return nb


def build_runner_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            block(
                """
                # Run All Augmentation Notebooks

                Batch runner for all `augment_*.ipynb` notebooks in this folder.
                """
            )
        ),
        nbf.v4.new_code_cell(RUNNER_NOTEBOOK_CODE),
        nbf.v4.new_code_cell(RUNNER_EXECUTION_CODE),
    ]
    return nb


def build_visualization_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            block(
                """
                # Visualize Augmentation Results

                This notebook is for understanding augmentation outcomes, not just inspecting saved CSV files.

                It loads saved augmentation artifacts across datasets and variants, then visualizes:
                - baseline vs augmented metrics
                - augmentation gains by dataset and model
                - class counts at each major stage
                - synthetic kind mix per emotion
                - per-class precision, recall, F1, and support
                - best-model confusion matrices
                """
            )
        ),
        nbf.v4.new_code_cell(VIS_NOTEBOOK_IMPORTS),
        nbf.v4.new_code_cell(VIS_HELPERS),
        nbf.v4.new_code_cell(VIS_SUMMARY),
        nbf.v4.new_code_cell(VIS_OVERVIEW_PLOTS),
        nbf.v4.new_code_cell(VIS_MODEL_DELTA),
        nbf.v4.new_code_cell(VIS_PAIRWISE),
        nbf.v4.new_code_cell(VIS_FOCUS),
    ]
    return nb


def main() -> None:
    AUGMENTATION_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_key in DATASET_KEYS:
        notebook = build_augmentation_notebook(dataset_key)
        write_notebook(AUGMENTATION_DIR / f"augment_{dataset_key}.ipynb", notebook)

    write_notebook(AUGMENTATION_DIR / "run_all_augmentation.ipynb", build_runner_notebook())
    write_notebook(AUGMENTATION_DIR / "visualize_augmentation_results.ipynb", build_visualization_notebook())

    print(f"Generated notebooks in {AUGMENTATION_DIR}")


if __name__ == "__main__":
    main()
