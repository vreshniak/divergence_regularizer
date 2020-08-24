import torch


def exact_jacobian(output, input, create_graph=True):
	jacobian = []

	Id = torch.zeros(*output.shape).to(input.device)
	for i in range(input.numel()):
		Id.data.flatten()[i] = 1.0

		jac_i = torch.autograd.grad(
			outputs=output,
			inputs=input,
			grad_outputs=Id,
			create_graph=create_graph,  # need create_graph to find it's derivative
			only_inputs=True)[0]

		jacobian.append(jac_i)

		Id.data.flatten()[i] = 0.0
	return torch.stack(jacobian, dim=0).reshape(output.shape+input.shape)


class DivergenceReg(torch.nn.Module):
	def __init__(self, n=1):
		self.n = n
		super(DivergenceReg, self).__init__()


	def forward(self, output, input, create_graph=True, method='rnd'):
		'''
		Compute divergence averaged over batch dimension
		'''
		batch_dim = input.size()[0]

		if method=='rnd':
			# Randomized version based on Hutchinson algorithm for trace evaluation
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
			reg = reg / self.n / batch_dim

		elif method=='exact':
			jac = exact_jacobian(output, input, create_graph).reshape((batch_dim,output.numel()//batch_dim,batch_dim,input.numel()//batch_dim))
			# divergence of each batch dimension
			reg = torch.stack([ jac[i,:,i,:].diag().sum() for i in range(batch_dim) ], dim=0)
			# total divergence
			reg = reg.sum() / batch_dim

		return reg