class NetWriter():
  
  def __init__(self):
    self.proto = ""
    self.header_template = """name: "Isingnet" layer {type: 'Python' name: 'ising_data' top: 'data' top: 'theta' python_param { module: 'ising_data_layer' layer: 'IsingDataLayer' param_str: '{ "batch_size":%d, "beta":%f, "lattice_size":%d, "gibbs_steps":%d}'}}"""
    self.conv_template = """\nlayer { bottom: "%s" top: "%s" name: "%s" type: "Convolution" convolution_param { num_output: %d kernel_size: %d stride: 1 pad: 1 weight_filler { type: "gaussian" std: 0.1 } bias_filler { type: "constant" value: 0 } }}"""
    self.batchnorm_template = """\nlayer { bottom: "%s" top: "%s" name: "%s" type: "BatchNorm" batch_norm_param { use_global_stats: %s } }\nlayer { bottom: "%s" top: "%s" name: "%s" type: "Scale" scale_param { bias_term: true } }"""
    self.relu_template = """\nlayer { bottom: "%s" top: "%s" name: "%s" type: "ReLU"}"""
    self.fc_template = """\nlayer { bottom: "%s" top: "%s" name: "%s" type: "InnerProduct" inner_product_param { num_output: %d weight_filler { type: "gaussian" std: 0.1 } bias_filler { type: "constant" value: 0 } } }"""
    self.loss_template = """\nlayer { name: "loss" type: "EuclideanLoss" bottom: "%s" bottom: "theta" top: "loss"}"""
    self.num_conv=0
    self.num_fc=0
    self.bottom=""

  def header(self,batch_size=64,beta=0.4406,lattice_size=10,gibbs_steps=100):
    self.proto += self.header_template %(batch_size,beta,lattice_size,gibbs_steps)
    self.bottom="data"

  def conv(self,num_filters,kernel_size):
    self.num_conv +=1
    name = "conv%d" % (self.num_conv)
    self.proto += self.conv_template %(self.bottom,name,name,num_filters,kernel_size)
    self.bottom=name

  def batchnorm(self,use_global):
    name="bn_%s" % self.bottom
    name2="scale_%s" % self.bottom
    self.proto += self.batchnorm_template %(self.bottom,self.bottom,name,str(use_global),self.bottom,self.bottom,name2)

  def relu(self):
    name="relu_%s" % self.bottom
    self.proto += self.relu_template %(self.bottom,self.bottom,name)

  def fc(self,num_out):
    self.num_fc +=1
    name = "fc%d" % (self.num_fc)
    self.proto += self.fc_template %(self.bottom, name, name, num_out)
    self.bottom=name

  def loss(self):
    self.proto += self.loss_template % (self.bottom)
