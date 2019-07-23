import seaborn as sns
from mmcv import load
import os
import matplotlib.pyplot as plt
import numpy as np

dir_path = os.path.dirname(os.getcwd())

ASPECT_RATIO_PATH = dir_path + '/image_analysis/711_aspect_ratio_distribution.pickle'
DIMENSION_PATH = dir_path + '/image_analysis/711_dimension_distribution.pickle'
BBOX_ASPECT_RATIO_PATH = dir_path + '/image_analysis/711_bbox_aspect_ratio_distribution.pickle'
BBOX_RELATIVE_RATIO_PATH = dir_path + '/image_analysis/711_bbox_relative_ratio_distribution.pickle'
BBOX_RELATIVE_DIMENSION_PATH = dir_path + '/image_analysis/711_bbox_relative_dimension_distribution.pickle'

# image aspect ratio
x = load(ASPECT_RATIO_PATH)
sns.distplot(x, bins=100, kde=False, rug=False, axlabel='image aspect ratio (width / height)').set_title('image aspect ratio')
plt.show()
print('image aspect ratio stats')
print('mean: {}, median: {}, total number: {}, smallest: {}, largest: {}'.format(np.mean(x), np.median(x), len(x), min(x), max(x)))
print('5%: {}, 95%: {}'.format(np.percentile(x, 5), np.percentile(x, 95)))
print('')

# image dimension
p = load(DIMENSION_PATH)
sns.jointplot(x=[pair[0] for pair in p], y=[pair[1] for pair in p]).set_axis_labels("width", "height")
plt.show()
print('image dimension')
print('mean: {}; {}, median: {}; {}, total number: {}, smallest: {}; {}, largest: {}; {}'.format(
        np.mean([pair[0] for pair in p]),
        np.mean([pair[1] for pair in p]),
        np.median([pair[0] for pair in p]),
        np.median([pair[1] for pair in p]),
        len(p),
        min([pair[0] for pair in p]),
        min([pair[1] for pair in p]),
        max([pair[0] for pair in p]),
        max([pair[1] for pair in p])
    )
)
print('5%: {}; {}, 95%: {}; {}'.format(
        np.percentile([pair[0] for pair in p], 5),
        np.percentile([pair[1] for pair in p], 5),
        np.percentile([pair[0] for pair in p], 95),
        np.percentile([pair[1] for pair in p], 95)
        )
)
print('') 

# bbox aspect ratio
y = load(BBOX_ASPECT_RATIO_PATH)
fig, axs = plt.subplots(nrows=2, ncols=3)
sns.distplot(y['Periodontal_disease'], bins=10, kde=False, rug=False, axlabel='bbox aspect ratio (width / height)', ax=axs[0, 0], hist_kws={"range": [0,5]}).set_title('Periodontal_disease')
sns.distplot(y['abscess'], bins=10, kde=False, rug=False, axlabel='bbox aspect ratio (width / height)', ax=axs[0, 1], hist_kws={"range": [0,5]}).set_title('abscess')
sns.distplot(y['caries'], bins=10, kde=False, rug=False, axlabel='bbox aspect ratio (width / height)', ax=axs[0, 2], hist_kws={"range": [0,5]}).set_title('caries')
sns.distplot(y['wedge_shaped_defect'], bins=10, kde=False, rug=False, axlabel='bbox aspect ratio (width / height)', ax=axs[1, 0], hist_kws={"range": [0,5]}).set_title('wedge_shaped_defect')
sns.distplot(y['calculus'], bins=10, kde=False, rug=False, axlabel='bbox aspect ratio (width / height)', ax=axs[1, 1], hist_kws={"range": [0,5]}).set_title('calculus')
sns.distplot(y['overall'], bins=10, kde=False, rug=False, axlabel='bbox aspect ratio (width / height)', ax=axs[1, 2], hist_kws={"range": [0,5]}).set_title('overall')
plt.show()
print('bbox aspect ratio stats')
print('Periodontal_disease, mean: {}, median: {}, total number: {}, smallest: {}, largest: {}'.format(np.mean(y['Periodontal_disease']), np.median(y['Periodontal_disease']), len(y['Periodontal_disease']), min(y['Periodontal_disease']), max(y['Periodontal_disease'])))
print('5%: {}, 95%: {}'.format(np.percentile(y['Periodontal_disease'], 5), np.percentile(y['Periodontal_disease'], 95)))
print('abscess, mean: {}, median: {}, total number: {}, smallest: {}, largest: {}'.format(np.mean(y['abscess']), np.median(y['abscess']), len(y['abscess']), min(y['abscess']), max(y['abscess'])))
print('5%: {}, 95%: {}'.format(np.percentile(y['abscess'], 5), np.percentile(y['abscess'], 95)))
print('caries, mean: {}, median: {}, total number: {}, smallest: {}, largest: {}'.format(np.mean(y['caries']), np.median(y['caries']), len(y['caries']), min(y['caries']), max(y['caries'])))
print('5%: {}, 95%: {}'.format(np.percentile(y['caries'], 5), np.percentile(y['caries'], 95)))
print('wedge_shaped_defect, mean: {}, median: {}, total number: {}, smallest: {}, largest: {}'.format(np.mean(y['wedge_shaped_defect']), np.median(y['wedge_shaped_defect']), len(y['wedge_shaped_defect']), min(y['wedge_shaped_defect']), max(y['wedge_shaped_defect'])))
print('5%: {}, 95%: {}'.format(np.percentile(y['wedge_shaped_defect'], 5), np.percentile(y['wedge_shaped_defect'], 95)))
print('calculus, mean: {}, median: {}, total number: {}, smallest: {}, largest: {}'.format(np.mean(y['calculus']), np.median(y['calculus']), len(y['calculus']), min(y['calculus']), max(y['calculus'])))
print('5%: {}, 95%: {}'.format(np.percentile(y['calculus'], 5), np.percentile(y['calculus'], 95)))
print('overall, mean: {}, median: {}, total number: {}, smallest: {}, largest: {}'.format(np.mean(y['overall']), np.median(y['overall']), len(y['overall']), min(y['overall']), max(y['overall'])))
print('5%: {}, 95%: {}'.format(np.percentile(y['overall'], 5), np.percentile(y['overall'], 95)))
print('')

# bbox relative aspect ratio
z = load(BBOX_RELATIVE_RATIO_PATH)
fig, axs = plt.subplots(nrows=2, ncols=3)
sns.distplot(z['Periodontal_disease'], bins=10, kde=False, rug=False, axlabel='bbox relative aspect ratio', ax=axs[0, 0], hist_kws={"range": [0,5]}).set_title('Periodontal_disease')
sns.distplot(z['abscess'], bins=10, kde=False, rug=False, axlabel='bbox relative aspect ratio', ax=axs[0, 1], hist_kws={"range": [0,5]}).set_title('abscess')
sns.distplot(z['caries'], bins=10, kde=False, rug=False, axlabel='bbox relative aspect ratio', ax=axs[0, 2], hist_kws={"range": [0,5]}).set_title('caries')
sns.distplot(z['wedge_shaped_defect'], bins=10, kde=False, rug=False, axlabel='bbox relative aspect ratio', ax=axs[1, 0], hist_kws={"range": [0,5]}).set_title('wedge_shaped_defect')
sns.distplot(z['calculus'], bins=10, kde=False, rug=False, axlabel='bbox relative aspect ratio', ax=axs[1, 1], hist_kws={"range": [0,5]}).set_title('calculus')
sns.distplot(z['overall'], bins=10, kde=False, rug=False, axlabel='bbox relative aspect ratio', ax=axs[1, 2], hist_kws={"range": [0,5]}).set_title('overall')
plt.show()

print('bbox relative aspect ratio stats')
print('Periodontal_disease, mean: {}, median: {}, total number: {}, smallest: {}, largest: {}'.format(np.mean(z['Periodontal_disease']), np.median(z['Periodontal_disease']), len(z['Periodontal_disease']), min(z['Periodontal_disease']), max(z['Periodontal_disease'])))
print('5%: {}, 95%: {}'.format(np.percentile(z['Periodontal_disease'], 5), np.percentile(z['Periodontal_disease'], 95)))
print('abscess, mean: {}, median: {}, total number: {}, smallest: {}, largest: {}'.format(np.mean(z['abscess']), np.median(z['abscess']), len(z['abscess']), min(z['abscess']), max(z['abscess'])))
print('5%: {}, 95%: {}'.format(np.percentile(z['abscess'], 5), np.percentile(z['abscess'], 95)))
print('caries, mean: {}, median: {}, total number: {}, smallest: {}, largest: {}'.format(np.mean(z['caries']), np.median(z['caries']), len(z['caries']), min(z['caries']), max(z['caries'])))
print('5%: {}, 95%: {}'.format(np.percentile(z['caries'], 5), np.percentile(z['caries'], 95)))
print('wedge_shaped_defect, mean: {}, median: {}, total number: {}, smallest: {}, largest: {}'.format(np.mean(z['wedge_shaped_defect']), np.median(z['wedge_shaped_defect']), len(z['wedge_shaped_defect']), min(z['wedge_shaped_defect']), max(z['wedge_shaped_defect'])))
print('5%: {}, 95%: {}'.format(np.percentile(z['wedge_shaped_defect'], 5), np.percentile(z['wedge_shaped_defect'], 95)))
print('calculus, mean: {}, median: {}, total number: {}, smallest: {}, largest: {}'.format(np.mean(z['calculus']), np.median(z['calculus']), len(z['calculus']), min(z['calculus']), max(z['calculus'])))
print('5%: {}, 95%: {}'.format(np.percentile(z['calculus'], 5), np.percentile(z['calculus'], 95)))
print('overall, mean: {}, median: {}, total number: {}, smallest: {}, largest: {}'.format(np.mean(z['overall']), np.median(z['overall']), len(z['overall']), min(z['overall']), max(z['overall'])))
print('5%: {}, 95%: {}'.format(np.percentile(z['overall'], 5), np.percentile(z['overall'], 95)))
print('')

# bbox relative dimension
q = load(BBOX_RELATIVE_DIMENSION_PATH)
sns.jointplot(x=[pair[0] for pair in q['Periodontal_disease']], y=[pair[1] for pair in q['Periodontal_disease']]).set_axis_labels("Periodontal_disease relative width", "Periodontal_disease relative height")
sns.jointplot(x=[pair[0] for pair in q['abscess']], y=[pair[1] for pair in q['abscess']]).set_axis_labels("abscess relative width", "abscess relative height")
sns.jointplot(x=[pair[0] for pair in q['caries']], y=[pair[1] for pair in q['caries']]).set_axis_labels("caries relative width", "caries relative height")
sns.jointplot(x=[pair[0] for pair in q['wedge_shaped_defect']], y=[pair[1] for pair in q['wedge_shaped_defect']]).set_axis_labels("wedge_shaped_defect relative width", "wedge_shaped_defect relative height")
sns.jointplot(x=[pair[0] for pair in q['calculus']], y=[pair[1] for pair in q['calculus']]).set_axis_labels("calculus relative width", "calculus relative height")
sns.jointplot(x=[pair[0] for pair in q['overall']], y=[pair[1] for pair in q['overall']]).set_axis_labels("overall relative width", "overall relative height")
plt.show()

print('bbox relative dimension')
print('Periodontal_disease, mean: {}; {}, median: {}; {}, total number: {}, smallest: {}; {}, largest: {}; {}'.format(
        np.mean([pair[0] for pair in q['Periodontal_disease']]),
        np.mean([pair[1] for pair in q['Periodontal_disease']]),
        np.median([pair[0] for pair in q['Periodontal_disease']]),
        np.median([pair[1] for pair in q['Periodontal_disease']]),
        len(q['Periodontal_disease']),
        min([pair[0] for pair in q['Periodontal_disease']]),
        min([pair[1] for pair in q['Periodontal_disease']]),
        max([pair[0] for pair in q['Periodontal_disease']]),
        max([pair[1] for pair in q['Periodontal_disease']])
    )
)
print('5%: {}; {}, 95%: {}; {}'.format(
        np.percentile([pair[0] for pair in q['Periodontal_disease']], 5),
        np.percentile([pair[1] for pair in q['Periodontal_disease']], 5),
        np.percentile([pair[0] for pair in q['Periodontal_disease']], 95),
        np.percentile([pair[1] for pair in q['Periodontal_disease']], 95)
        )
)
print('abscess, mean: {}; {}, median: {}; {}, total number: {}, smallest: {}; {}, largest: {}; {}'.format(
        np.mean([pair[0] for pair in q['abscess']]),
        np.mean([pair[1] for pair in q['abscess']]),
        np.median([pair[0] for pair in q['abscess']]),
        np.median([pair[1] for pair in q['abscess']]),
        len(q['abscess']),
        min([pair[0] for pair in q['abscess']]),
        min([pair[1] for pair in q['abscess']]),
        max([pair[0] for pair in q['abscess']]),
        max([pair[1] for pair in q['abscess']])
    )
)
print('5%: {}; {}, 95%: {}; {}'.format(
        np.percentile([pair[0] for pair in q['abscess']], 5),
        np.percentile([pair[1] for pair in q['abscess']], 5),
        np.percentile([pair[0] for pair in q['abscess']], 95),
        np.percentile([pair[1] for pair in q['abscess']], 95)
        )
)
print('caries, mean: {}; {}, median: {}; {}, total number: {}, smallest: {}; {}, largest: {}; {}'.format(
        np.mean([pair[0] for pair in q['caries']]),
        np.mean([pair[1] for pair in q['caries']]),
        np.median([pair[0] for pair in q['caries']]),
        np.median([pair[1] for pair in q['caries']]),
        len(q['caries']),
        min([pair[0] for pair in q['caries']]),
        min([pair[1] for pair in q['caries']]),
        max([pair[0] for pair in q['caries']]),
        max([pair[1] for pair in q['caries']])
    )
)
print('5%: {}; {}, 95%: {}; {}'.format(
        np.percentile([pair[0] for pair in q['caries']], 5),
        np.percentile([pair[1] for pair in q['caries']], 5),
        np.percentile([pair[0] for pair in q['caries']], 95),
        np.percentile([pair[1] for pair in q['caries']], 95)
        )
)
print('wedge_shaped_defect, mean: {}; {}, median: {}; {}, total number: {}, smallest: {}; {}, largest: {}; {}'.format(
        np.mean([pair[0] for pair in q['wedge_shaped_defect']]),
        np.mean([pair[1] for pair in q['wedge_shaped_defect']]),
        np.median([pair[0] for pair in q['wedge_shaped_defect']]),
        np.median([pair[1] for pair in q['wedge_shaped_defect']]),
        len(q['wedge_shaped_defect']),
        min([pair[0] for pair in q['wedge_shaped_defect']]),
        min([pair[1] for pair in q['wedge_shaped_defect']]),
        max([pair[0] for pair in q['wedge_shaped_defect']]),
        max([pair[1] for pair in q['wedge_shaped_defect']])
    )
)
print('5%: {}; {}, 95%: {}; {}'.format(
        np.percentile([pair[0] for pair in q['wedge_shaped_defect']], 5),
        np.percentile([pair[1] for pair in q['wedge_shaped_defect']], 5),
        np.percentile([pair[0] for pair in q['wedge_shaped_defect']], 95),
        np.percentile([pair[1] for pair in q['wedge_shaped_defect']], 95)
        )
)
print('calculus, mean: {}; {}, median: {}; {}, total number: {}, smallest: {}; {}, largest: {}; {}'.format(
        np.mean([pair[0] for pair in q['calculus']]),
        np.mean([pair[1] for pair in q['calculus']]),
        np.median([pair[0] for pair in q['calculus']]),
        np.median([pair[1] for pair in q['calculus']]),
        len(q['calculus']),
        min([pair[0] for pair in q['calculus']]),
        min([pair[1] for pair in q['calculus']]),
        max([pair[0] for pair in q['calculus']]),
        max([pair[1] for pair in q['calculus']])
    )
)
print('5%: {}; {}, 95%: {}; {}'.format(
        np.percentile([pair[0] for pair in q['calculus']], 5),
        np.percentile([pair[1] for pair in q['calculus']], 5),
        np.percentile([pair[0] for pair in q['calculus']], 95),
        np.percentile([pair[1] for pair in q['calculus']], 95)
        )
)
print('overall, mean: {}; {}, median: {}; {}, total number: {}, smallest: {}; {}, largest: {}; {}'.format(
        np.mean([pair[0] for pair in q['overall']]),
        np.mean([pair[1] for pair in q['overall']]),
        np.median([pair[0] for pair in q['overall']]),
        np.median([pair[1] for pair in q['overall']]),
        len(q['overall']),
        min([pair[0] for pair in q['overall']]),
        min([pair[1] for pair in q['overall']]),
        max([pair[0] for pair in q['overall']]),
        max([pair[1] for pair in q['overall']])
    )
)
print('5%: {}; {}, 95%: {}; {}'.format(
        np.percentile([pair[0] for pair in q['overall']], 5),
        np.percentile([pair[1] for pair in q['overall']], 5),
        np.percentile([pair[0] for pair in q['overall']], 95),
        np.percentile([pair[1] for pair in q['overall']], 95)
        )
)
print('')
