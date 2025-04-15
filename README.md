# Mastet thesis

### Embedding class percentage constraints and feature dependency conditions into CTAB-GAN+ for tabular data generation

## Abstact 

Data generation via conditional Generative Adversarial Networks (GANs) has gained significant attention. Unlike image or other types of data, tabular data synthesis contains unique challenges due to heterogeneous structure and data formats. The evolution of models involves enhancing the quality of synthetic samples, handling class imbalance and bias, addressing issues during training, selecting appropriate evaluation metrics, and  incorporating domain-specific knowledge through constraints and conditions.

The complexity of such generative models often increases, when auxiliary modules are added to preserve feature dependencies, patterns and statistical properties of the original dataset. In conditional tabular GANs a conditional vector guides the training process by assigning a single non-zero value to indicate the chosen category or mode of a categorical or continuous variables.

This study investigates the integration of class balance constraints and two-column conditions into conditional tabular GAN framework and explores their combinations in various model configurations. To implement these ideas, the construction of the conditional vector was modified and adjusted for each experimental scenario. An additional loss term was introduced to regulate constraints. Data transformation, sampling, activa tion functions of discriminator and generator were altered to adapt the novel conditional vector design.

The model and synthetic data were evaluated using both quantitative metrics and qualitative approaches, supported by graphical visualizations. The results demonstrate an ability of the model to preserve data characteristics. While the conditional vector is effective in producing targeted distributions, the constraint loss is incapable to regulate user-defined class balance percentages. Nevertheless, the outcomes illustrate the potential of incorporating conditions and constraints separately through the conditional vector and still successfully maintaining original data characteristics.