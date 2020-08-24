import torch


class divreg(torch.nn.Module):
	def __init__(self, n=1):
		self.n = n
		super(divreg, self).__init__()


	def forward(self, output, input, create_graph=True):
		batch_dim = input.size()[0]
		reg = 0
		for _ in range(self.n):
			v = torch.randn_like(input)
			jac_v, = torch.autograd.grad(
				outputs=output,
				inputs=input,
				grad_outputs=v,
				create_graph=create_graph,  # need create_graph to find it's derivative
				only_inputs=True)
			reg = reg + (v*jac_v).sum()
		return reg / self.n / batch_dim