from manipulate.transform import Transform


class SequentialModel(Transform):
  def __init__(self, transforms):
    super().__init__()
    self.transforms = transforms

  def __call__(self, tokens, idx):
    new_tokens = tokens[:]
    for transform in self.transforms:
      new_tokens_ = transform(new_tokens, idx)
      if new_tokens_ is not None:
        new_tokens = new_tokens_
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def multi_ptr_trans(self, tokens, indices):
    new_tokens = tokens[:]
    for transform in self.transforms:
      new_tokens_ = transform.multi_ptr_trans(new_tokens, indices)
      if new_tokens_ is not None:
        new_tokens = new_tokens_
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def __str__(self) -> str:
    return '\t'.join([str(transform) for transform in self.transforms])
