̅
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2	"
adj_xbool( "
adj_ybool( "
grad_xbool( "
grad_ybool( 
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.18.02v2.18.0-rc2-4-g6550e4bd8028��
�
ConstConst*#
_output_shapes
:� *
dtype0*��
value��B��� "��      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?      �?�jW?@Q
?�}?�X?>9�>jNs?:#5>��{?vu�=��~?�6f=h�?\�=<�?B��<��?W�#<��?\D�;��? >O;��? �:��?m�:��?j:��?_˥9��?Sw:9  �?��h?3վ
�f?���>T?�{N?�G�>��o?�oK>��z?���=�a~?�p�=�|?\�=��?>ԣ<��?�C8<��?�=�;��?i;��?i;��?j�:��?_�%:��?Sw�9��?Á>&p}��E~?����	P?W/?m/?)m\?mN�>�t?��+>�]|?���=O�~?MiZ=â? ��<��?�1�<��?n<�?�Ϯ;�?��D;��?�:��?�x:��?~�:��?ϽA�0U'��IG?�� ��t?�.�>�'?��A?�a�>��k?�ed>	�y?�.>i�}?ȍ�=GZ?��#=��?�@�<l�?�<O<��?��;X�?X�;z�?j;��?\˥:��?Rw::��?|u�,<�>�ڥ>@2r��?�r)���F?�L!?Dw�>@�`?��>|�u?<!>��|?!۵=�~?��L=�?�M�<�?���<��?��<i�?�֣;.�?�D8;��?1>�:��?&i:��?�����u?t�k�Sy�Mxr?o?���0`?4/�>i�?2IS?�}�>�q?o A>�g{?��=Q�~?ќu=�?�,
=��?Fl�<4�?�.<E�?Y��;��?];��?��:��?}ً:��?F0(?��@?n�6�`O3�o�L?�t�D�r?���>s�$?��C?�`�>�kl?P�`>��y?�F�=�~?2>�=�_?`1!=;�?�R�<��?*�K<��?�_�;e�?���;~�?�;��?f(�:��?�F}?����-z� Y�}?�Q��1}?�,>��7?_[2?Ҥ�>R�f?R&�>�w?�/>�i}?Q��=f.?�48=��?�8�<�?%i<^�?<��?�i�;V�?T�%;��?Ow�:��?2�>�?i�F�p��>��>��t�/�?:���H?�!?�4�>K�_?.>��u?�/#>u�|??�=��~?�6O=�?��<v�?x�<��?:t<Y�?+ץ;)�?��:;��?8��:��?�D�d�V�l��FJ?�p�����k�z?
�R��jW?@Q
?�}?�X?>9�>jNs?:#5>��{?vu�=��~?�6f=h�?\�=<�?B��<��?W�#<��?\D�;��? >O;��? �:��?\��r�;�8ǽ0�~?!A����q�;m?o��*&d?�=�>�r?��P?���>�p?�G>,{?r��=�s~?�4}=��?ks=[�?�;�<v�?j84<	�?���;��?��c;��?2 ;��?�\	��X?3��>�d?&��UOK�*xX?���n?���>��?5�G?,��>��m??�X>�1z?�+�=�(~?��=�j?e=��?x̮<�?q�D<H�?��;��?�x;��?w�;��?" �>oNh?[?pe?LS������<?��,���v?���>��*?�>?@��>ëj?%�j>�0y?о> �}?���=�P?iV(=��?�\�<}�?k�T<v�?ً�;@�?%��;r�?�;��?r�}?�>l�?������u����U?�|K�oF|?�.>�[5?í4?�V�>�Qg?O|>}x?d�>�~}?h�=5?FG5=ɿ?��<��?V^e<��?|� <��?��;\�?^(#;��?Dy&?�zB���U?���u���"�<���>p�c��[?�ސ=A:??03*?���>h�c?D��>O�v?O> }?勬=�?�7B=I�?3}�<��?3�u<��?
3
<��?�n�;C�?��.;��?h��,)u�60�>�h�.�p�0>�>���>J�t��?�3��}H?�.?6$�>��_?��>#�u?P$#>��|?��=��~?�'O="�?�<z�? �<��?�i<Y�?1˥;)�?Cw:;��?%v�6⌾m�	�~�}���I�O�?(�=D4~���}?���Q?��?�?��[?Co�>dt?#>->IO|?�|�=��~?�\=S�?Ҝ�<�?�A�<��?�<�?�'�;�?�F;��?�@@��
)?Eg$��;D�Ũ��T?��r�Č�Ny?��h�Y?�?��	?��W?�>;�r?�S7>.�{?���=��~?�i=ܕ?+=n�?�r�<b�?��%<��?���;��?&�Q;��?iy>�}?R�s�����A���dv?^�p���x��@r?6���S_`?'��>p�?\*S?緩>��q?:dA>�d{?�f�=H�~?��u=��?�]
=��?~��<,�? /<B�?4��;��?�m];��?ȶi?"��>K�w�x�~>;g)=��?�jξ�Ej���h?3վ
�f?���>T?�{N?�G�>��o?�oK>��z?���=�a~?�p�=�|?\�=��?>ԣ<��?�C8<��?�=�;��?i;��?/V?�7��/��:?z6�>7�o?��pRT�J�\?�=���l?t��>
�?��I?�ɺ>'[n?�vU><`z??J�=�7~?>�=�o?��=O�?��<��?zA<n�?���;��?x�t;��?�o��GF��'{?'�?yH?-S2���7���N?���q?���>�$?��D?<�>\�l?�w_>j�y?��=~?[]�=xa?4 =��?�5�<$�?��J<��?+��;j�?�1�;��?�X��g���>��n?�V?t?� P��;���>?�*�g6v?F8�>5*?�8??͞�>j�j?�ri>3By?�>��}?Ӕ=�R?@{'=1�?>f�<��?��S<��?�S�;D�?��;t�?��g��-�>�/M?
?�w?�ۅ>g��+ܾ(�,?��<���y?��`>+0?c�9?��>s	i?Qhs>��x?&�	>�}?yH�=YC?H�.=S�?і�< �?j]<�?r��;�?bً;g�?8��x�}?~5?J�=d�?W�S�2�v��x5?�M���|?#(>��5?�4?2�>�g?wW}>�
x?S|>ny}?r��=O3?,	6=?�?W��<��?�Sf<��?��<��?��;Z�?7C?��%?��b?��n?�)��k�~�B�����?][�$}~?�2�=E�;?�<.?Qa�>e?��>Uew?V/>�D}? 2�=�"?�O==��?���<��?5�o<��?ڴ<��?Ѐ�;M�?�t?8���-A ?*�]��bF?��!���~� ��=���>Fqg�ʥ?��V=��@?�8(?~�> c?��>��v?$�>}? ��=D?��D=y�?9(�<�?��x<r�?)�<��?�T�;?�?*��>Wmv�������l;
?�xW�k�v����>���>|5q���?�_v�u1F?	"?���>��`?}~�>�v?�� >I�|?��=D�~?��K=Ǯ?�X�<P�?v��<��?w<n�?<(�;0�?��)���?����KS�`��>o�w�,g�+8�>��t>��x��?�u��7K? �?{}�>ݴ^?Th�>xOu?�@&>y�|?��=��~?5#S=�?��<r�?���<I�?�?<?�?���;!�?��|��>�j��d˾�~�ہ���O�?A?Á>&p}��E~?����	P?W/?m/?)m\?mN�>�t?��+>�]|?���=O�~?MiZ=â? ��<��?�1�<��?n<�?�Ϯ;�?�ξ�,j?4�|���>2����m�VN2�N�7?|P*=R���3|?�/���T?^�?�?�Z?�0�>"�s?\�1>�|?�q�=Z�~?9�a=r�?M��<��?�̎<�?\� <��?\��;�?M*?��U?�A�(?��#���D� �5VT?�o�?���Uy?&h�Y?�?��	?j�W?��>s?oF7>��{?���=��~?��h=�?�=s�?h�<d�?��%<��?w�;��?9�?��Y�!���Z$u?��X�
���^ξ�Hj?�!�y�|���u?돾�;]?�� ?��?�+U?��>�/r?�<>̚{?�T�={�~?�:p=0�?�$=R�?+�<��?��*<n�?�J�;��?�q?�;Y�;��>ոv?ymx�'7w�%up���x?Iւ��w��>q?�N��B2a?�>��?؝R?Ҿ�>_Xq?�B>�U{?���=��~?�w=@�?�<= �?F��<�?7'0<6�?y�;��?�:۾_Xg��R=?�P,?ST�$�=*_r�*�?ܙ����o��l?�'ƾ��d?d"�>�>?��O?M��>�zp?�>H>�{?�5�= o~?�~=�?�T=��?^9�<O�?~U5<��?-��;��?8�}����
|?.\3>	�l�Z�>�]�=x3~?
����e��f?$`��ph?��>K�?OM?J]�>�o?p�M>��z?'��=�W~?��=�y?�l=��?sԥ<��?Ã:<��?���;��?�$�-�C?0#m?Y���7�B�6�%?y��>R�t?W��Y�tp_?���g�k?μ�>��?��J?�%�>�n?��S>�zz?�=�?~?Y��=1r?v�=%�?�o�<��?�?<��?���;��?���>�t?�2?EP����HZ?���>\�c?����|J�MX?�M	���n?m��>p ?׽G?J�>��m?(Y>d-z?^��=f'~?�I�=mj?M�=��?�
�<
�?K�D<E�?Dm�;��?K�v?l��>�t=�����l��y?3[?�xK?q0���9�IP?';���q?�>6�#?��D?��>��l?��^>9�y?	��=;~?��=tb?�=+�?���<=�?�J<�?�@�;m�?��>?t�*�%���bZ`� B�=�?�<?��,?ϽA�0U'��IG?�� ��t?�.�>�'?��A?�a�>��k?�ed>	�y?�.>i�}?ȍ�=GZ?��#=��?�@�<l�?�<O<��?��;X�?�m"�D�|�E�_��"��3��>��k?�{X?=�?�zQ��'�F�=?�+��nv?(��>��*?M�>?��>n�j?�j>�9y?�d>��}?�/�=�Q?��'=��?�ۼ<��?kT<~�?X��;B�?(�j���̾ԗ���f=8�'?cEA?�=m?�b�>�_�w����3?�	6�C�x?@v>��-?9�;?���>2�i?�o>��x?��>Ҿ}?}ј=LI??�+=5�?�v�<��?J�Y<8�?��;,�?1�T��?��P�{?�[?*{?΃z?��R>ډj��5;�z)?��?�ORz?6{V>	:1?�8?p�>��h?F4u>[�x?�
>�}?(s�=@?�0=l�?��<��?��^<��?���;�?��<��?w�¾��l?�y?�b>b�?�<2�s��Z���m?�I�G�{?D�6>�{4?��5?��>Νg?��z>4x?2>��}?��=~7?u*4=��?���<��?��c<��?�1 <��?��Y?�{?}�/>2|?K�~?�[���0}?QG�j?z���W�/�?��Q��5}?��>��7?�P2?ͳ�>��f?a/�>��w?�9>�i}?��=G.?�A8=��?�G�<�?�#i<]�?�<��?��f?mFݾ��+?d�=?o�j?��˾�r?����b~�9����?4�Y�qF~?~V�==�:?b/?+M�>�ce?t��>�{w?an>�K}?dW�=�$?}Y<=��?s��<�?2Rn<�?b<��?��=�	~�4|v?XJ�>D�??��)�G-`?;�������J����>��`�? �=f�=?ث+?���>�=d?x��>Kw?g�>}-}?���=<?�p@=��?]}�<�?h�s<��?9�<��?V�D���#�eu?_ב��3?]�k�F?.R!����2�=�+�>�[g�P�?��X=0�@?hD(?Rn�>c?g��>�v?�>y}?���=g?S�D=��?B�<!�?��x<s�?�<��?")t�D�>*�(?<�@��DX>�9z��'?T�A��{��>>U��>'m�)�?�P�<[�C?X�$?���>��a?;M�>�Wv?H	>��|?i:�=]?��H=W�?"��<�?��}<"�?��<��?�U��w?a >7�|��oӽ͡~��)?�p\�|u�,<�>�ڥ>@2r��?�r)���F?�L!?Dw�>@�`?��>|�u?<!>��|?!۵=�~?��L=�?�M�<�?���<��?��<i�?$�+?��=?��ɾdHk�l�оy�i�Q;�>��o�1m�L��>^�>nyv�M�?��+��I?|�?^��>/l_?zՐ>:�u?�n$>��|?�{�=��~?2�P=ʪ?���<�?��<z�?��<N�?��|?�&���R�C�e�+���=��5>��{�*b�k��>��\>"�y��N?�����fL?A!?g�>f)^?ۗ�>�!u?��'>�|? �=�~?b�T=j�?���<��?���<$�?f�<3�?���>�k����t(�T^�������׸  ��{U�z�?�V$>��|���~?׽"O?�x?�j?��\?	Y�>��t?�*>�i|?f��=$�~?��X=��?t�<��?�J�<��?;j<�?���MT���]���>k�z���M��=5�R�{���E�;{"?ϊ�=i�~�m�}?G����Q?��?k�?̒[? �>�It?A.>�F|?�\�=�~?�]=y�?<��<��?��<r�?T<��?���,C�<s��8b?WX~�^~�=�T��L�o�J�4�bk5?�uG=@��R`|?E�+�RTT?�?��??Z?�כ>[�s?�31>D#|?w��=��~?�*a=�?�S�<��?�x�<�?�= <��?^���lZ?�֙=�F?�h��o�>;5��i\���!�|�F?f_��&����z?P�K���V?�6?��?��X?1��>2is?/d4>�{?B��=P�~?�Ae=D�?^� =��?�<��?�'#<��?�P�>�]f?�_?��M?X�;�s�-?�$'���A������U?>z���z��(y?B%k��GY?�^?S%
?ӆW?_Q�>�r?��7>?�{?�;�=��~?yXi=��?�D=V�?��<[�?�&<��?�,~?��=	�n?̖�>5��
�_?��F��G!�V���h�b?�$���(~��,w?E��2�[?]|?LK?k"V??�>��r?D�:>Ŵ{?Y��=��~?Qom=͑?�=%�?&>�<��?_�(<��?C#?�eE�S{?��B�F�C�FJ{?�3`�d#���l��:om?��3�	|���t?EՔ���]?J�>�m?��T?�Ŧ>�	r?�=>��{?�z�=��~?�q=��?k�=��?7՘<��?2�+<e�?#��V�s�ϧ:?�3/�|��=�~?m�r��룾�����u?t�k�Sy�Mxr?o?���0`?4/�>i�?2IS?�}�>�q?o A>�g{?��=Q�~?ќu=�?�,
=��?Fl�<4�?�.<E�?Sw��%�� �>Ľw��<�>_�g?�2}���ψ:�{�{?�Ñ��gu���o?���Zb?,�>q�?t�Q?�4�>�q?HND>{@{?���=�~~?z�y=�?z=r�?U�<��?׸1<%�?�:=��j,?R
���s���/?�):?������<�*��b?����p�n�l?��¾Hqd?��>��?WZP?>�>��p?�{G>n{?uW�=r~?�}=�?^�=,�?b��<h�?��4<�?K_+>�c|?f�C�$%���`?n��>�z�pS>N��<��?��Ǿ�k���i?rѾvf?{��>1�?��N?A��>p?u�J>��z?	��=(e~?M��=�}?�=��?n1�<��?z�7<��?Ԇk?���>��}�`W�@�{?M9>x8m��{�>Y��=^A~?��a�e�(,f?��hh?E��>��?&VM?�P�>G�o?��M>f�z?m��=X~?���=�y?�a=��?zȥ<��?Kv:<��?�S?���PXi�Д�>��}?�C	��tX���?jH\>�z?������^���b?<��QGj?�d�>�?&�K?��>�o?� Q>k�z?�2�=�J~?��=�u?6�=:�?�_�<(�?`=<��?΁ټ���:F�1}U?Kpf?߾��<���,?Y��>qBs?�
�ŋW�E�^?�����l?b�>:�?�<J?���>ӑn?�+T>�qz?���=%=~?��=Pq?x�=��?���<��?�I@<|�?[[����93����?�U8?;�1�zP���K?}H�>j?�JqO�؅Z?[���m?��>��?��H?�_�>2n?�VW>�Fz?vn�=e/~?�=�l?�I=}�?���<J�?�3C<Y�?��e�mZ�>�?|�[?%��>�(b��龅�c?m��>Ր^?o!��F�|2V?�2��ro?_�>�� ?G?R�>��m?��Z>�z?�=p!~?(�=�h?�=�?�$�<��?�F<5�?M��N~?�nd?� �>B�.>U?|��w��B�t?�?\�P?�X,��K=�K�Q?���=q?��>�"?apE?a��>+�l?H�]>�y?���=G~?,3�=d?+�=��?���<f�?[I<�?�F?5!"?��~?|2�����3R}�q��5~?F0(?��@?n�6�`O3�o�L?�t�D�r?���>s�$?��C?�`�>�kl?P�`>��y?�F�=�~?2>�=�_?`1!=;�?�R�<��?*�K<��?Qxs?:8��6�J?�6�C���Ee��6s=^�?��:?s>/?x�@���(��G?�����s?#R�>��&?$B?��>v�k?��c>�y?�� >V�}?.I�=�Z?�~#=��?��<{�?��N<��?#��>��w�X��>�Jp�ى3��|6���p>W�x?�-K?�?��I�ϫ���B?�&�OGu?���>o�(?Wv@?��>?Mk?�#g>�dy?F�>��}?!T�=4V?��%=J�?���<�?��Q<��?�@-� w<���U�U]z�cc�J6�2w�>=Cj?7�Y?�?�9R� �Ds=?-,��v?�߉>��*?��>?�S�>�j?�Jj>�5y?��>��}?
_�=uQ?�(=��?��<��?��T<z�?�2|���/>�2�WT7���|�n>$�I?�NT?�f?F��>�Z�6
�j�7?v2��w?��>"�,?G=?���>'j?�qm>�y?�\>c�}?�i�=�L?f*=D�?���<�?`�W<S�?Q�ƾ��k?��x�-To���|�q9>X2?�7?� p?/z�>�Ja��#�{02?D�7���x?B�p>�.?�O;?��>.�i?��p>l�x?�*>��}?�t�=�G?6�,=��?�E�<��?-�Z<,�?n�?aS?{r�?/�>>d�5��>�P?I6?��w?!��>��g�nپ�I,?Y=�[�y?��^>(]0?ڎ9?�7�>_�h?"�s>j�x?��	>e�}?��=�B?V /=(�?���<�?�k]<�?��?Z���/_!�ݻF? �4��j5?�!g?��>��|?
>i�m����J7&?N�B���z?��L>�52?��7?���>�_h?��v>�rx?��>��}?L��=�=?sM1=��?�s�<��?�U`<��?�?��[�߁��C+~?�~�R�d?��v?��>��?\ ]=��r�[���?��G�ް{?\;>
4?n�5?�q�>�g?*z>{@x?i�>��}?��=�8?��3=��?�
�<�?�?c<��?�b�Z^e��P�>"Sg?���}?�~?���=��?�l<���v�r����?
�L��{|?�()>��5?5/4?0�>�&g?�)}>�x?b>\z}?���=�3?��5=W�?���<��?Z)f<��?}o~���OX?�<	?�(>��|?8�~?�+���F}?����-z� Y�}?�Q��1}?�,>��7?_[2?Ҥ�>R�f?R&�>�w?�/>�i}?Q��=f.?�48=��?�8�<�?%i<^�?W@!���F?��?w�a;�4�>U�b?��v?����Jx?�cy���|��� �eS?V���}?/$>�j9?��0?�;�>!�e?s��>ҥw?��>PY}?败= )?��:=�?���<��?��k<4�?�_�>s?#Y?t��JF7?T�2?Eg?YDܾ��p?����ð~�E�νb|?�qZ��`~?�!�=`,;?�.?���>Ce??H�>�pw?;�>|H}?t��=�#?��<=V�?zf�<��?��n<�?��w?r�>��>>�f�S�e?d��>��O?�F���f?m�ܾɽ��8�e�> �^���~?���=�<?��,?�c�>+�d?�؄>�;w?c�>r7}?�ɩ=d?�?=��?n��<m�?��q<��?\�;?.��|潥_~�{}?�M>I2? �7���Z?�������8<C��>�qb��=?��=�>?��*?��>i�c?�h�>tw?md>5&}?kԫ=�?�hA=�?`��<��?I�t<��?+M4� |����a�G��|?�3���?�YT��iL?t��j��(�=�b�>f���?�r=�S@?��(?���>�Nc?���>��v?V1>�}?�ޭ=e?еC=&�?Q+�<T�?�w<��?�gl�tľ��q�녧�5�a?���*R�>eKj�s<?��-��
~�N��=H��>��i���?N�)=�B?$'?	�>]�b?��>a�v? �>}?6�=�?�F=`�?@��<��?؍z<V�?�bR�(�?�Py�vh>��0?9��Zp>�x�Z�)?N�?�i�{��N7>���>��l���?��<�C?Z%?���>�a?�>e_v?��>@�|?��=#?�OH=��?.Y�<3�?�w}<(�?��=��?K�3��6?���>
�f�q�q=���"�?ˣO��x��o>�ȳ>ݲo���? ��;%OE?V#?>'�>�Ia?ӥ�>�&v?Q�>1�|?���=i?��J=Ư?��<��?�0�<��?�-\?��?��\���y?�>x�}�,��2~��B ?)�]�d!u�ɛ�>S��>�kr�b�?�B�-�F?&!!?��>�`?/4�>��u?�c!>��|?�=��~?��L=�?��<�?���<��?��d?�i�SE�>�p?��=���{�_���Y�t�2�>�?i�F�p��>��>��t�/�?:���H?�!?�4�>K�_?.>��u?�/#>u�|??�=��~?�6O=�?��<v�?x�<��?��=ގ~�Z�I?��?S����]`����G�c��`�>�r�pLk�ֲ�>U��>�"w�F�?�;B��J?q?>��>�4_?�O�>(yu?#�$>ȧ|?c�=��~?m�Q=6�?״�<��?Z��<k�?�G��^ �ղ~?�>�=��:��.��`�ztK�A=d>Z�y�*Be�V��>�k>�y��t?6z����K??�9�>\^?ݒ>>u?%�&>�|?{&�=��~?N�S=P�?�K�<D�?<�<;�?��r���>9e?��nh��3ؾ��<���,����={~�Y~^�q=�><L>��z�_?�˩��6M?�?u��>1�]?�i�>Du?�(>с|?�0�=��~?*V=f�?���<��?y�<
�?V#{�}.x?�&?�Z��*~����]X�������<���uW���
?I,>�Y|�n�~?�ν!�N?Y�?|?9]?q��>��t?�_*>�n|?�:�=��~?jX=v�?�y�<�?��<��?=�.?��:?Q{�G���{�c�G>@m�V���虽�F���N�p�?�H>�}��3~?�B� <P?/�?2Y?xT\?���>݈t?Z+,>
[|?zD�=~�~?նZ=��?h�<n�?�b�<��?h�{?��8����)uV�u_�+�>2�z�>�R���2��|�2F�
."?mI�=��~���}?.1���Q?5�?ڕ?�[?J�>9Kt?��->WG|?`N�=>�~?�]=��?H��<��?�׋<s�?�[�>x�l��h���վ��,�/�<?����6�5$���]v���<�b-?�ɗ=�K��|?�5��+S?x�?r�?��Z?���>�t?!�/>q3|?9X�=��~?nP_=��?&>�<,�?�L�<@�?(���Q���}�Y> dӾv)i?�/}��a>'���k0n�6�2��b7?F.=����9|?x-/�z�T?�?�?�Z?�$�>�s?N�1>V|?b�=��~?3�a=��?��<��?���<�?��8#=y�D��#?$�߽_x~?�r�G�>%B��c�P(��"A?k33<��Vh{?A��V?�v?dE?�WY?��>��s?WX3>|?�k�=�~?��c=w�?�5 =��?b6�<��?.����\?Wf���ps?�NR>�z?*`��F�>�D�d�V�l��FJ?�p�����k�z?
�R��jW?@Q
?�}?�X?>9�>jNs?:#5>��{?vu�=��~?�6f=h�?\�=<�?B��<��?�o�>nZd?*{>.x?���>��]?+�F�kW!?� �Q�G��K���R?�'V�]��8�y?�d���X?(?��	?��W?� >�s?��6>��{?�=�~?h�h=S�?��=��?! �<n�?��~?*�=�q9?�{0?��>?� +?�'���A?�*3��6�8�4�Z?$���N��yx?Ppv��#Z?P�?�
?�W?JL�>F�r?��8>��{?���=a�~?�j=9�?3=��? ��<8�?/|?I@H���z?�I>>j?s��>�#�t\?�D�3$��q񾋿a?~ ��J~�NWw?_	���w[?+�? ?�>V?1գ>E�r? �:>��{?;��=��~?�m=�?�c==�?�	�<�?�����er�O9o?�K���~?!~�=�.��X�o?o�S������׾--h?4*��;}�� v?�ό���\?��?�S? tU?�]�>�Gr?KM<>l�{?���=�~?qio=�?�=��?�~�<��? ux���v���?��L�Q�y?&�\��4���{?�3a��y� ?����m?�6�:�{�N�t?���s^?���>��?ϧT?��>fr?o>>�{?$��=�~?�q=ʍ?n�=��?��<��?&:���/?:�=]#�Go\?�+���W9  �?;l�KžH5���r?��U�xZz�xs?:���R_?qM�>2�?��S?^m�>��q?k�?>'w{?���=*�~?�t=��?�E	=/�?zh�<\�?m7=>q�{?���0
c��)?�D@�TX5> �{?�t��#��I�����v?�pu���x�r?Aܦ�6�`?���>��?
S?���>|q??�A>6a{?շ�=2�~?MOv=f�?=�
=}�?Wݛ<#�?�Cm?�B�>�]�p�?��>YLk�@a�>��o?�{�k�F��\U��az?`��)yv�s�p?�p����a?�Q�>�?�8R?U{�>�6q?�tC>K{?��=*�~?�x=,�?��=��?4R�<��?Q?j�������=�g�=g�~�
;?Mf\?��~�4pý�����|?H䙾�)t�e�n?u�����b?���>�E?deQ?��>:�p?p>E>�4{?L��={~?q�z=�?(=�?Ǟ<��?�75�Կ���S�G�?��f��fy��)'?�A?\��r�;�8ǽ0�~?!A����q�;m?o��*&d?�=�>�r?��P?���>�p?�G>,{?r��=�s~?�4}=��?ks=[�?�;�<v�?>S]��� ��;��j?�q��[�+�F?xB!? �~��y�=��(�e�?�r����n�h{k?�Ⱦ�Le?���>X�?��O?��>�cp?��H>k{?���=�l~?��=_�?ξ=��?ʰ�<<�?��c��t�>�g>l}?�B��'�7`?��>d�z���O>�]v<��?�tǾ��k���i?�.Ѿ�lf?��>��?��N?���>_p?�J>v�z?���=ce~? �=~?0
=��?�%�<�?Oǽ��~?pe'?ůA?$Tl���ľ��r?*ߣ>�:t�9x�>Aב=�Y?4D־:�h�,�g?�uپ��g?(s�>��?�N?��>0�o?�bL>N�z?���=^~?=�=�{?�U=*�?���<��?z�H?�?��t?p8�>�6�L���3}?E�>Yk��w�>c6>��}?���R e���e?��4�h?���>�?F,M?���>d�o?�+N>��z?u��=�V~?x3�=ay?�=k�?]�<��?r?�Ц���v?�ㆾ��x��Kq>��?�����`��s�>�;>ٰ{?l:��Da��c?����i?D%�>�@?OL?0�>�Ao?�O>`�z?P �=O~?�Y�=w?P�=��?7��<L�?]Wr>D�x���,?Z�<��Y��?�z?�%S��R�n{?�as>�x?�� ��O]���a?��񾜳j?;v�>8f?(pK?U��>��n?}�Q>��z?	�=�G~?��=�t?�7=��?��<�?�0�o^9�S�6>��{��%�#�C?�5m?���[C���%?Ys�>��t?ԛ��!Y��r_?i���P�k?���>t�?��J?$�>S�n?��S>�zz?��=�?~?��=4r?�=&�?�m�<��?8a{�%�A>�A���wm�f쿾vUm?�pX?X��܊1�Cn8?ἰ>�Cp?�g�[�T�'1]?S� ���l?��>m�?r�I?5��>bn?�LU>vbz?��=D8~?Ḟ=�o?g�=a�?��<��?�(��L�m?T�O���#,���h?k�<?��,�<>��;I?`w�>��j?��| P���Z?����m?4J�>�?��H?�'�>:n?�W>Jz? #�=�0~?s�=Qm?�=��?�W�<T�?�?*nP?�|�d���E�{>g'x?K?؄K��\	��X?3��>�d?&��UOK�*xX?���n?���>��?5�G?,��>��m??�X>�1z?�+�=�(~?��=�j?e=��?x̮<�?[�?~SG���`��
�>q�?SJX?��>��c�18�(�d?=��>^?��!�$JF�JV?�}���o?���>�!?(�F?�)�>�|m?��Z>�z?)4�=� ~?�>�=Yh?v�=�?QA�<��?UW�>s�]�X���z_?XkE?4�"?�j�>:�t��i���o?}�?^�V?(�0A�yS?OF��hp?V�>s,"?F?2��>/m?k\>��y?�<�=�~?�d�=�e?��=<�?)��<��?xx��Qc���X=+�?IPn?�>ݼ�=�5~�QƆ���v?D�?ON?P .�Ш;���P? ��Fq?5!�>�H#?;*E?�)�>��l?92^>��y?�D�=�~?	��=Kc?%G=n�?+�<S�?X�~�s����?@Q?�?�l=T�s����$�)�w|?`�"?�vE?��3�g6��5N?����r?�K�>d$?_>D??��>ґl?1�_>)�y?=M�=�~?'��=�`?{� =��?ٟ�<�?Դ���I?�wl?�'�>|}w?�邾��p���x��Ӈ��o?��-?��;?��9�dG0��zK?�X���r?�q�>�}%?�PC?(�>RBl?��a>��y?wU�=� ~?B׏=(^?��!=��?��<��?���>�q?pX|?Dh,���V?:-���ξ�@j�R�	=��?>8?e�1?�?�AR*�9�H?���1�s?	��>y�&?�aB?K��>7�k?��c>Ǚy?Ѯ >w�}?[��=�[?$)#=��?���<��?I�x?��m>��>?�+�G� ?%G����JT��m>��}?��A?#N'?�hD��1$���E?�{"�}�t? ��>��'?IqA?$�>��k?Me>�y?ݲ>2�}?p#�=�X?wt$=(�?^��<J�?�8?�a1�	��>�>v���>�Do��\2�5�7��m>�y?��J?y&?��I������B?��%��@u?V̒>m�(?@?N��>1Pk?=g>�ey?�>��}?�I�=MV?ɿ%=R�?4s�<�?�F��){�,&���u���A=����P��0����>Y�q?cQS?n�?�dN��s��??j)�I�u?��>��)?c�??
�>G�j?I�h>/Ky?ݺ>w�}?�o�=�S?'=z�?
�<��?ln�����?��c)������v��$g��ܾ" �>oNh?[?pe?LS������<?��,���v?���>��*?�>?@��>ëj?%�j>�0y?о> �}?���=�P?iV(=��?�\�<}�?=�O���?u�|��#�2d�YiU���v��Ԇ��7?Oh\?Q3b?���>��W��
�D�9?�!0��Ww?��>��+?I�=?��>�Xj?�dl>�y?��>y�}?���=AN?��)=��?�Ѿ<8�?-nY=��?	�k�C&�>~�H������~��T��!�?iNN?��h?E��>��[�28���6?Jh3���w?�!~>�-?�<?��>�j?J*n>��x?��>��}?��=�K?�*=��?�F�<��?�t^?�_�>���JKR?S2p�����~�|a�=�h+?�$>?�9n?�t�>��_�rh���e3?U�6�.�x?3t>�.?�;?��>��i?��o>��x?t�>9�}?��=�H?Q8,=�?]��<��?��b?�z���'6�*�?2���o���v���>F�=?�,?�s?�a�>��c�� �H0?��9�H6y?->j>9)/?��:?ԅ�>�[i?��q>)�x?D�>�}?�-�=F?��-=,�?00�<e�?���=��~��r�>o^?Wv���>^g��P�>��M?�L?*=w?�̄>�@g�r�۾k�,?��<��y?JC`>F50?��9?c��>/i?�ys>��x?�	>��}?�S�=>C?��.=J�?��<�?�YJ�����a?;;�>�S�L�?�O�4L?�[?�?�z?]�Q>)�j���̾�g)?��?�VUz?�BV>�?1?O�8?gx�>�h?B>u>��x?��
>ۢ}?�y�=o@?/0=g�?��<��?�Hq���>lW?�̒�C��XaJ?�D2���7?��g?���>!}?��>L�m����%?�B�A�z?�<L>I2?e�7?���>^Yh?�w>�px?}�>�}?���=�=?ve1=��?���<��?��i�Ay?�<N?��� ��~q?����]T?
�q?�`�>��~?���=��p��Ю�y"?��E��Z{?51B>�P3?��6?�h�>h?�x>�Tx?)�>��}?�Ş=�:?��2=��?{�<D�?O22?�7?};�>�Dn��0ڼ��?�EξNj?|�x?��p>�?�N=FBs�c�����?Q�H���{?� 8>#W4?�5?&��>-�g?(�z>8x?��>�}?|�=�7?�3=��?Mx�<��?��z?��J�WY?��}{��4�>&Wu?�@p���x?r�}?�>l�?������u����U?�|K�oF|?�.>�[5?í4?�V�>�Qg?O|>}x?d�>�~}?h�=5?FG5=ɿ?��<��?��>[�n�Ď.�B;�w�?�pR?��q���?��?d3=�G?����<�w��q��?�?�7N���|?b�#>X_6?�3?3��>��f?�~>��w?��>�u}?Q7�=2?��6=ݾ?�a�<f�?�y�FO�|�w�������K?I�?���=�1~?�~?򘀽3�}?��V�y��ka�` ?��P�H}?��><a7?��2?�B�>�f?:�>��w?z�>`l}?6]�=+/?��7=�?���<�?���Ňk=?t��[�>�q?�!�>E��>a�t?�|?T&�a�{?T�>��`{���A��C?�zS�~w}?�>�a8?��1?���>�Df?�̀>p�w?��>c}?��=8,?)9=�?�K�<��?
�
dense_2/biasVarHandleOp*
_output_shapes
: *

debug_namedense_2/bias/*
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
�

dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
�
dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape
: *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

: *
dtype0
�
dense_5/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_5/kernel/*
dtype0*
shape
:@H*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:@H*
dtype0
�
dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape:	� *
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	� *
dtype0
�
dense_5/biasVarHandleOp*
_output_shapes
: *

debug_namedense_5/bias/*
dtype0*
shape:H*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:H*
dtype0
�
dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape:	 �*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	 �*
dtype0
�
multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention/value/bias/*
dtype0*
shape
:
 *0
shared_name!multi_head_attention/value/bias
�
3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

:
 *
dtype0
�
dense_4/biasVarHandleOp*
_output_shapes
: *

debug_namedense_4/bias/*
dtype0*
shape:@*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:@*
dtype0
�
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention/value/kernel/*
dtype0*
shape: 
 *2
shared_name#!multi_head_attention/value/kernel
�
5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*"
_output_shapes
: 
 *
dtype0
�
multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *.

debug_name multi_head_attention/key/bias/*
dtype0*
shape
:
 *.
shared_namemulti_head_attention/key/bias
�
1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

:
 *
dtype0
�
layer_normalization/betaVarHandleOp*
_output_shapes
: *)

debug_namelayer_normalization/beta/*
dtype0*
shape: *)
shared_namelayer_normalization/beta
�
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
: *
dtype0
�
multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention/query/bias/*
dtype0*
shape
:
 *0
shared_name!multi_head_attention/query/bias
�
3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

:
 *
dtype0
�
dense_3/biasVarHandleOp*
_output_shapes
: *

debug_namedense_3/bias/*
dtype0*
shape:@*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:@*
dtype0
�
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *;

debug_name-+multi_head_attention/attention_output/bias/*
dtype0*
shape: *;
shared_name,*multi_head_attention/attention_output/bias
�
>multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp*multi_head_attention/attention_output/bias*
_output_shapes
: *
dtype0
�
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention/query/kernel/*
dtype0*
shape: 
 *2
shared_name#!multi_head_attention/query/kernel
�
5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*"
_output_shapes
: 
 *
dtype0
�
dense_3/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_3/kernel/*
dtype0*
shape
: @*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

: @*
dtype0
�
layer_normalization/gammaVarHandleOp*
_output_shapes
: **

debug_namelayer_normalization/gamma/*
dtype0*
shape: **
shared_namelayer_normalization/gamma
�
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
: *
dtype0
�
dense_4/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_4/kernel/*
dtype0*
shape
:@@*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:@@*
dtype0
�
layer_normalization_1/betaVarHandleOp*
_output_shapes
: *+

debug_namelayer_normalization_1/beta/*
dtype0*
shape: *+
shared_namelayer_normalization_1/beta
�
.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
: *
dtype0
�
layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *,

debug_namelayer_normalization_1/gamma/*
dtype0*
shape: *,
shared_namelayer_normalization_1/gamma
�
/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
: *
dtype0
�
dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
�
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *=

debug_name/-multi_head_attention/attention_output/kernel/*
dtype0*
shape:
  *=
shared_name.,multi_head_attention/attention_output/kernel
�
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*"
_output_shapes
:
  *
dtype0
�
multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention/key/kernel/*
dtype0*
shape: 
 *0
shared_name!multi_head_attention/key/kernel
�
3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*"
_output_shapes
: 
 *
dtype0
�
dense_5/bias_1VarHandleOp*
_output_shapes
: *

debug_namedense_5/bias_1/*
dtype0*
shape:H*
shared_namedense_5/bias_1
m
"dense_5/bias_1/Read/ReadVariableOpReadVariableOpdense_5/bias_1*
_output_shapes
:H*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpdense_5/bias_1*
_class
loc:@Variable*
_output_shapes
:H*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:H*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:H*
dtype0
�
dense_5/kernel_1VarHandleOp*
_output_shapes
: *!

debug_namedense_5/kernel_1/*
dtype0*
shape
:@H*!
shared_namedense_5/kernel_1
u
$dense_5/kernel_1/Read/ReadVariableOpReadVariableOpdense_5/kernel_1*
_output_shapes

:@H*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpdense_5/kernel_1*
_class
loc:@Variable_1*
_output_shapes

:@H*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape
:@H*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:@H*
dtype0
�
%seed_generator_3/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_3/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_3/seed_generator_state
�
9seed_generator_3/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_3/seed_generator_state*
_output_shapes
:*
dtype0	
�
%Variable_2/Initializer/ReadVariableOpReadVariableOp%seed_generator_3/seed_generator_state*
_class
loc:@Variable_2*
_output_shapes
:*
dtype0	
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0	*
shape:*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0	
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0	
�
dense_4/bias_1VarHandleOp*
_output_shapes
: *

debug_namedense_4/bias_1/*
dtype0*
shape:@*
shared_namedense_4/bias_1
m
"dense_4/bias_1/Read/ReadVariableOpReadVariableOpdense_4/bias_1*
_output_shapes
:@*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpdense_4/bias_1*
_class
loc:@Variable_3*
_output_shapes
:@*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:@*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
e
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
:@*
dtype0
�
dense_4/kernel_1VarHandleOp*
_output_shapes
: *!

debug_namedense_4/kernel_1/*
dtype0*
shape
:@@*!
shared_namedense_4/kernel_1
u
$dense_4/kernel_1/Read/ReadVariableOpReadVariableOpdense_4/kernel_1*
_output_shapes

:@@*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpdense_4/kernel_1*
_class
loc:@Variable_4*
_output_shapes

:@@*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape
:@@*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
i
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes

:@@*
dtype0
�
%seed_generator_2/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_2/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_2/seed_generator_state
�
9seed_generator_2/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_2/seed_generator_state*
_output_shapes
:*
dtype0	
�
%Variable_5/Initializer/ReadVariableOpReadVariableOp%seed_generator_2/seed_generator_state*
_class
loc:@Variable_5*
_output_shapes
:*
dtype0	
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0	*
shape:*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0	
e
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes
:*
dtype0	
�
dense_3/bias_1VarHandleOp*
_output_shapes
: *

debug_namedense_3/bias_1/*
dtype0*
shape:@*
shared_namedense_3/bias_1
m
"dense_3/bias_1/Read/ReadVariableOpReadVariableOpdense_3/bias_1*
_output_shapes
:@*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOpdense_3/bias_1*
_class
loc:@Variable_6*
_output_shapes
:@*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:@*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
e
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
:@*
dtype0
�
dense_3/kernel_1VarHandleOp*
_output_shapes
: *!

debug_namedense_3/kernel_1/*
dtype0*
shape
: @*!
shared_namedense_3/kernel_1
u
$dense_3/kernel_1/Read/ReadVariableOpReadVariableOpdense_3/kernel_1*
_output_shapes

: @*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOpdense_3/kernel_1*
_class
loc:@Variable_7*
_output_shapes

: @*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape
: @*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
i
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes

: @*
dtype0
�
layer_normalization_1/beta_1VarHandleOp*
_output_shapes
: *-

debug_namelayer_normalization_1/beta_1/*
dtype0*
shape: *-
shared_namelayer_normalization_1/beta_1
�
0layer_normalization_1/beta_1/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta_1*
_output_shapes
: *
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOplayer_normalization_1/beta_1*
_class
loc:@Variable_8*
_output_shapes
: *
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape: *
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
e
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
: *
dtype0
�
layer_normalization_1/gamma_1VarHandleOp*
_output_shapes
: *.

debug_name layer_normalization_1/gamma_1/*
dtype0*
shape: *.
shared_namelayer_normalization_1/gamma_1
�
1layer_normalization_1/gamma_1/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma_1*
_output_shapes
: *
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOplayer_normalization_1/gamma_1*
_class
loc:@Variable_9*
_output_shapes
: *
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape: *
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
e
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes
: *
dtype0
�
%seed_generator_1/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_1/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_1/seed_generator_state
�
9seed_generator_1/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_1/seed_generator_state*
_output_shapes
:*
dtype0	
�
&Variable_10/Initializer/ReadVariableOpReadVariableOp%seed_generator_1/seed_generator_state*
_class
loc:@Variable_10*
_output_shapes
:*
dtype0	
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0	*
shape:*
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0	
g
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes
:*
dtype0	
�
dense_2/bias_1VarHandleOp*
_output_shapes
: *

debug_namedense_2/bias_1/*
dtype0*
shape: *
shared_namedense_2/bias_1
m
"dense_2/bias_1/Read/ReadVariableOpReadVariableOpdense_2/bias_1*
_output_shapes
: *
dtype0
�
&Variable_11/Initializer/ReadVariableOpReadVariableOpdense_2/bias_1*
_class
loc:@Variable_11*
_output_shapes
: *
dtype0
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0*
shape: *
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0
g
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes
: *
dtype0
�
dense_2/kernel_1VarHandleOp*
_output_shapes
: *!

debug_namedense_2/kernel_1/*
dtype0*
shape:	� *!
shared_namedense_2/kernel_1
v
$dense_2/kernel_1/Read/ReadVariableOpReadVariableOpdense_2/kernel_1*
_output_shapes
:	� *
dtype0
�
&Variable_12/Initializer/ReadVariableOpReadVariableOpdense_2/kernel_1*
_class
loc:@Variable_12*
_output_shapes
:	� *
dtype0
�
Variable_12VarHandleOp*
_class
loc:@Variable_12*
_output_shapes
: *

debug_nameVariable_12/*
dtype0*
shape:	� *
shared_nameVariable_12
g
,Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_12*
_output_shapes
: 
h
Variable_12/AssignAssignVariableOpVariable_12&Variable_12/Initializer/ReadVariableOp*
dtype0
l
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes
:	� *
dtype0
�
dense_1/bias_1VarHandleOp*
_output_shapes
: *

debug_namedense_1/bias_1/*
dtype0*
shape:�*
shared_namedense_1/bias_1
n
"dense_1/bias_1/Read/ReadVariableOpReadVariableOpdense_1/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_13/Initializer/ReadVariableOpReadVariableOpdense_1/bias_1*
_class
loc:@Variable_13*
_output_shapes	
:�*
dtype0
�
Variable_13VarHandleOp*
_class
loc:@Variable_13*
_output_shapes
: *

debug_nameVariable_13/*
dtype0*
shape:�*
shared_nameVariable_13
g
,Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_13*
_output_shapes
: 
h
Variable_13/AssignAssignVariableOpVariable_13&Variable_13/Initializer/ReadVariableOp*
dtype0
h
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes	
:�*
dtype0
�
dense_1/kernel_1VarHandleOp*
_output_shapes
: *!

debug_namedense_1/kernel_1/*
dtype0*
shape:	 �*!
shared_namedense_1/kernel_1
v
$dense_1/kernel_1/Read/ReadVariableOpReadVariableOpdense_1/kernel_1*
_output_shapes
:	 �*
dtype0
�
&Variable_14/Initializer/ReadVariableOpReadVariableOpdense_1/kernel_1*
_class
loc:@Variable_14*
_output_shapes
:	 �*
dtype0
�
Variable_14VarHandleOp*
_class
loc:@Variable_14*
_output_shapes
: *

debug_nameVariable_14/*
dtype0*
shape:	 �*
shared_nameVariable_14
g
,Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_14*
_output_shapes
: 
h
Variable_14/AssignAssignVariableOpVariable_14&Variable_14/Initializer/ReadVariableOp*
dtype0
l
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*
_output_shapes
:	 �*
dtype0
�
layer_normalization/beta_1VarHandleOp*
_output_shapes
: *+

debug_namelayer_normalization/beta_1/*
dtype0*
shape: *+
shared_namelayer_normalization/beta_1
�
.layer_normalization/beta_1/Read/ReadVariableOpReadVariableOplayer_normalization/beta_1*
_output_shapes
: *
dtype0
�
&Variable_15/Initializer/ReadVariableOpReadVariableOplayer_normalization/beta_1*
_class
loc:@Variable_15*
_output_shapes
: *
dtype0
�
Variable_15VarHandleOp*
_class
loc:@Variable_15*
_output_shapes
: *

debug_nameVariable_15/*
dtype0*
shape: *
shared_nameVariable_15
g
,Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_15*
_output_shapes
: 
h
Variable_15/AssignAssignVariableOpVariable_15&Variable_15/Initializer/ReadVariableOp*
dtype0
g
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15*
_output_shapes
: *
dtype0
�
layer_normalization/gamma_1VarHandleOp*
_output_shapes
: *,

debug_namelayer_normalization/gamma_1/*
dtype0*
shape: *,
shared_namelayer_normalization/gamma_1
�
/layer_normalization/gamma_1/Read/ReadVariableOpReadVariableOplayer_normalization/gamma_1*
_output_shapes
: *
dtype0
�
&Variable_16/Initializer/ReadVariableOpReadVariableOplayer_normalization/gamma_1*
_class
loc:@Variable_16*
_output_shapes
: *
dtype0
�
Variable_16VarHandleOp*
_class
loc:@Variable_16*
_output_shapes
: *

debug_nameVariable_16/*
dtype0*
shape: *
shared_nameVariable_16
g
,Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_16*
_output_shapes
: 
h
Variable_16/AssignAssignVariableOpVariable_16&Variable_16/Initializer/ReadVariableOp*
dtype0
g
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16*
_output_shapes
: *
dtype0
�
#seed_generator/seed_generator_stateVarHandleOp*
_output_shapes
: *4

debug_name&$seed_generator/seed_generator_state/*
dtype0	*
shape:*4
shared_name%#seed_generator/seed_generator_state
�
7seed_generator/seed_generator_state/Read/ReadVariableOpReadVariableOp#seed_generator/seed_generator_state*
_output_shapes
:*
dtype0	
�
&Variable_17/Initializer/ReadVariableOpReadVariableOp#seed_generator/seed_generator_state*
_class
loc:@Variable_17*
_output_shapes
:*
dtype0	
�
Variable_17VarHandleOp*
_class
loc:@Variable_17*
_output_shapes
: *

debug_nameVariable_17/*
dtype0	*
shape:*
shared_nameVariable_17
g
,Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_17*
_output_shapes
: 
h
Variable_17/AssignAssignVariableOpVariable_17&Variable_17/Initializer/ReadVariableOp*
dtype0	
g
Variable_17/Read/ReadVariableOpReadVariableOpVariable_17*
_output_shapes
:*
dtype0	
�
,multi_head_attention/attention_output/bias_1VarHandleOp*
_output_shapes
: *=

debug_name/-multi_head_attention/attention_output/bias_1/*
dtype0*
shape: *=
shared_name.,multi_head_attention/attention_output/bias_1
�
@multi_head_attention/attention_output/bias_1/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/bias_1*
_output_shapes
: *
dtype0
�
&Variable_18/Initializer/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/bias_1*
_class
loc:@Variable_18*
_output_shapes
: *
dtype0
�
Variable_18VarHandleOp*
_class
loc:@Variable_18*
_output_shapes
: *

debug_nameVariable_18/*
dtype0*
shape: *
shared_nameVariable_18
g
,Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_18*
_output_shapes
: 
h
Variable_18/AssignAssignVariableOpVariable_18&Variable_18/Initializer/ReadVariableOp*
dtype0
g
Variable_18/Read/ReadVariableOpReadVariableOpVariable_18*
_output_shapes
: *
dtype0
�
.multi_head_attention/attention_output/kernel_1VarHandleOp*
_output_shapes
: *?

debug_name1/multi_head_attention/attention_output/kernel_1/*
dtype0*
shape:
  *?
shared_name0.multi_head_attention/attention_output/kernel_1
�
Bmulti_head_attention/attention_output/kernel_1/Read/ReadVariableOpReadVariableOp.multi_head_attention/attention_output/kernel_1*"
_output_shapes
:
  *
dtype0
�
&Variable_19/Initializer/ReadVariableOpReadVariableOp.multi_head_attention/attention_output/kernel_1*
_class
loc:@Variable_19*"
_output_shapes
:
  *
dtype0
�
Variable_19VarHandleOp*
_class
loc:@Variable_19*
_output_shapes
: *

debug_nameVariable_19/*
dtype0*
shape:
  *
shared_nameVariable_19
g
,Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_19*
_output_shapes
: 
h
Variable_19/AssignAssignVariableOpVariable_19&Variable_19/Initializer/ReadVariableOp*
dtype0
o
Variable_19/Read/ReadVariableOpReadVariableOpVariable_19*"
_output_shapes
:
  *
dtype0
�
!multi_head_attention/value/bias_1VarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention/value/bias_1/*
dtype0*
shape
:
 *2
shared_name#!multi_head_attention/value/bias_1
�
5multi_head_attention/value/bias_1/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/bias_1*
_output_shapes

:
 *
dtype0
�
&Variable_20/Initializer/ReadVariableOpReadVariableOp!multi_head_attention/value/bias_1*
_class
loc:@Variable_20*
_output_shapes

:
 *
dtype0
�
Variable_20VarHandleOp*
_class
loc:@Variable_20*
_output_shapes
: *

debug_nameVariable_20/*
dtype0*
shape
:
 *
shared_nameVariable_20
g
,Variable_20/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_20*
_output_shapes
: 
h
Variable_20/AssignAssignVariableOpVariable_20&Variable_20/Initializer/ReadVariableOp*
dtype0
k
Variable_20/Read/ReadVariableOpReadVariableOpVariable_20*
_output_shapes

:
 *
dtype0
�
#multi_head_attention/value/kernel_1VarHandleOp*
_output_shapes
: *4

debug_name&$multi_head_attention/value/kernel_1/*
dtype0*
shape: 
 *4
shared_name%#multi_head_attention/value/kernel_1
�
7multi_head_attention/value/kernel_1/Read/ReadVariableOpReadVariableOp#multi_head_attention/value/kernel_1*"
_output_shapes
: 
 *
dtype0
�
&Variable_21/Initializer/ReadVariableOpReadVariableOp#multi_head_attention/value/kernel_1*
_class
loc:@Variable_21*"
_output_shapes
: 
 *
dtype0
�
Variable_21VarHandleOp*
_class
loc:@Variable_21*
_output_shapes
: *

debug_nameVariable_21/*
dtype0*
shape: 
 *
shared_nameVariable_21
g
,Variable_21/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_21*
_output_shapes
: 
h
Variable_21/AssignAssignVariableOpVariable_21&Variable_21/Initializer/ReadVariableOp*
dtype0
o
Variable_21/Read/ReadVariableOpReadVariableOpVariable_21*"
_output_shapes
: 
 *
dtype0
�
multi_head_attention/key/bias_1VarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention/key/bias_1/*
dtype0*
shape
:
 *0
shared_name!multi_head_attention/key/bias_1
�
3multi_head_attention/key/bias_1/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias_1*
_output_shapes

:
 *
dtype0
�
&Variable_22/Initializer/ReadVariableOpReadVariableOpmulti_head_attention/key/bias_1*
_class
loc:@Variable_22*
_output_shapes

:
 *
dtype0
�
Variable_22VarHandleOp*
_class
loc:@Variable_22*
_output_shapes
: *

debug_nameVariable_22/*
dtype0*
shape
:
 *
shared_nameVariable_22
g
,Variable_22/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_22*
_output_shapes
: 
h
Variable_22/AssignAssignVariableOpVariable_22&Variable_22/Initializer/ReadVariableOp*
dtype0
k
Variable_22/Read/ReadVariableOpReadVariableOpVariable_22*
_output_shapes

:
 *
dtype0
�
!multi_head_attention/key/kernel_1VarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention/key/kernel_1/*
dtype0*
shape: 
 *2
shared_name#!multi_head_attention/key/kernel_1
�
5multi_head_attention/key/kernel_1/Read/ReadVariableOpReadVariableOp!multi_head_attention/key/kernel_1*"
_output_shapes
: 
 *
dtype0
�
&Variable_23/Initializer/ReadVariableOpReadVariableOp!multi_head_attention/key/kernel_1*
_class
loc:@Variable_23*"
_output_shapes
: 
 *
dtype0
�
Variable_23VarHandleOp*
_class
loc:@Variable_23*
_output_shapes
: *

debug_nameVariable_23/*
dtype0*
shape: 
 *
shared_nameVariable_23
g
,Variable_23/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_23*
_output_shapes
: 
h
Variable_23/AssignAssignVariableOpVariable_23&Variable_23/Initializer/ReadVariableOp*
dtype0
o
Variable_23/Read/ReadVariableOpReadVariableOpVariable_23*"
_output_shapes
: 
 *
dtype0
�
!multi_head_attention/query/bias_1VarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention/query/bias_1/*
dtype0*
shape
:
 *2
shared_name#!multi_head_attention/query/bias_1
�
5multi_head_attention/query/bias_1/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/bias_1*
_output_shapes

:
 *
dtype0
�
&Variable_24/Initializer/ReadVariableOpReadVariableOp!multi_head_attention/query/bias_1*
_class
loc:@Variable_24*
_output_shapes

:
 *
dtype0
�
Variable_24VarHandleOp*
_class
loc:@Variable_24*
_output_shapes
: *

debug_nameVariable_24/*
dtype0*
shape
:
 *
shared_nameVariable_24
g
,Variable_24/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_24*
_output_shapes
: 
h
Variable_24/AssignAssignVariableOpVariable_24&Variable_24/Initializer/ReadVariableOp*
dtype0
k
Variable_24/Read/ReadVariableOpReadVariableOpVariable_24*
_output_shapes

:
 *
dtype0
�
#multi_head_attention/query/kernel_1VarHandleOp*
_output_shapes
: *4

debug_name&$multi_head_attention/query/kernel_1/*
dtype0*
shape: 
 *4
shared_name%#multi_head_attention/query/kernel_1
�
7multi_head_attention/query/kernel_1/Read/ReadVariableOpReadVariableOp#multi_head_attention/query/kernel_1*"
_output_shapes
: 
 *
dtype0
�
&Variable_25/Initializer/ReadVariableOpReadVariableOp#multi_head_attention/query/kernel_1*
_class
loc:@Variable_25*"
_output_shapes
: 
 *
dtype0
�
Variable_25VarHandleOp*
_class
loc:@Variable_25*
_output_shapes
: *

debug_nameVariable_25/*
dtype0*
shape: 
 *
shared_nameVariable_25
g
,Variable_25/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_25*
_output_shapes
: 
h
Variable_25/AssignAssignVariableOpVariable_25&Variable_25/Initializer/ReadVariableOp*
dtype0
o
Variable_25/Read/ReadVariableOpReadVariableOpVariable_25*"
_output_shapes
: 
 *
dtype0
�
dense/bias_1VarHandleOp*
_output_shapes
: *

debug_namedense/bias_1/*
dtype0*
shape: *
shared_namedense/bias_1
i
 dense/bias_1/Read/ReadVariableOpReadVariableOpdense/bias_1*
_output_shapes
: *
dtype0
�
&Variable_26/Initializer/ReadVariableOpReadVariableOpdense/bias_1*
_class
loc:@Variable_26*
_output_shapes
: *
dtype0
�
Variable_26VarHandleOp*
_class
loc:@Variable_26*
_output_shapes
: *

debug_nameVariable_26/*
dtype0*
shape: *
shared_nameVariable_26
g
,Variable_26/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_26*
_output_shapes
: 
h
Variable_26/AssignAssignVariableOpVariable_26&Variable_26/Initializer/ReadVariableOp*
dtype0
g
Variable_26/Read/ReadVariableOpReadVariableOpVariable_26*
_output_shapes
: *
dtype0
�
dense/kernel_1VarHandleOp*
_output_shapes
: *

debug_namedense/kernel_1/*
dtype0*
shape
: *
shared_namedense/kernel_1
q
"dense/kernel_1/Read/ReadVariableOpReadVariableOpdense/kernel_1*
_output_shapes

: *
dtype0
�
&Variable_27/Initializer/ReadVariableOpReadVariableOpdense/kernel_1*
_class
loc:@Variable_27*
_output_shapes

: *
dtype0
�
Variable_27VarHandleOp*
_class
loc:@Variable_27*
_output_shapes
: *

debug_nameVariable_27/*
dtype0*
shape
: *
shared_nameVariable_27
g
,Variable_27/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_27*
_output_shapes
: 
h
Variable_27/AssignAssignVariableOpVariable_27&Variable_27/Initializer/ReadVariableOp*
dtype0
k
Variable_27/Read/ReadVariableOpReadVariableOpVariable_27*
_output_shapes

: *
dtype0
�
serve_keras_tensorPlaceholder*4
_output_shapes"
 :������������������*
dtype0*)
shape :������������������
�
StatefulPartitionedCallStatefulPartitionedCallserve_keras_tensordense/kernel_1dense/bias_1Const#multi_head_attention/query/kernel_1!multi_head_attention/query/bias_1!multi_head_attention/key/kernel_1multi_head_attention/key/bias_1#multi_head_attention/value/kernel_1!multi_head_attention/value/bias_1.multi_head_attention/attention_output/kernel_1,multi_head_attention/attention_output/bias_1layer_normalization/gamma_1layer_normalization/beta_1dense_1/kernel_1dense_1/bias_1dense_2/kernel_1dense_2/bias_1layer_normalization_1/gamma_1layer_normalization_1/beta_1dense_3/kernel_1dense_3/bias_1dense_4/kernel_1dense_4/bias_1dense_5/kernel_1dense_5/bias_1*%
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*:
_read_only_resource_inputs
	
*5
config_proto%#

CPU

GPU2*0J 8� �J *4
f/R-
+__inference_signature_wrapper___call___2143
�
serving_default_keras_tensorPlaceholder*4
_output_shapes"
 :������������������*
dtype0*)
shape :������������������
�
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_keras_tensordense/kernel_1dense/bias_1Const#multi_head_attention/query/kernel_1!multi_head_attention/query/bias_1!multi_head_attention/key/kernel_1multi_head_attention/key/bias_1#multi_head_attention/value/kernel_1!multi_head_attention/value/bias_1.multi_head_attention/attention_output/kernel_1,multi_head_attention/attention_output/bias_1layer_normalization/gamma_1layer_normalization/beta_1dense_1/kernel_1dense_1/bias_1dense_2/kernel_1dense_2/bias_1layer_normalization_1/gamma_1layer_normalization_1/beta_1dense_3/kernel_1dense_3/bias_1dense_4/kernel_1dense_4/bias_1dense_5/kernel_1dense_5/bias_1*%
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*:
_read_only_resource_inputs
	
*5
config_proto%#

CPU

GPU2*0J 8� �J *4
f/R-
+__inference_signature_wrapper___call___2198

NoOpNoOp
�*
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*�)
value�)B�) B�)
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures*
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27*
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
 21
"22
#23*
 
0
1
2
!3*
�
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719
820
921
:22
;23*
* 

<trace_0* 
"
	=serve
>serving_default* 
KE
VARIABLE_VALUEVariable_27&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_26&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_25&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_24&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_23&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_22&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_21&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_20&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_19&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_18&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_17'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_16'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_15'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_14'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_13'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_12'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_11'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_10'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_9'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_8'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_7'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_6'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_5'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_4'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_3'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_2'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_1'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEVariable'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE!multi_head_attention/key/kernel_1+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE.multi_head_attention/attention_output/kernel_1+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense_1/bias_1+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUElayer_normalization_1/gamma_1+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUElayer_normalization_1/beta_1+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_4/kernel_1+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUElayer_normalization/gamma_1+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_3/kernel_1+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE#multi_head_attention/query/kernel_1+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE,multi_head_attention/attention_output/bias_1+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdense_3/bias_1,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE!multi_head_attention/query/bias_1,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUElayer_normalization/beta_1,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEmulti_head_attention/key/bias_1,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE#multi_head_attention/value/kernel_1,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdense_4/bias_1,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE!multi_head_attention/value/bias_1,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEdense_1/kernel_1,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdense_5/bias_1,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEdense_2/kernel_1,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEdense_5/kernel_1,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdense/kernel_1,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEdense/bias_1,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdense_2/bias_1,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUE*

?	capture_2* 

?	capture_2* 

?	capture_2* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable!multi_head_attention/key/kernel_1.multi_head_attention/attention_output/kernel_1dense_1/bias_1layer_normalization_1/gamma_1layer_normalization_1/beta_1dense_4/kernel_1layer_normalization/gamma_1dense_3/kernel_1#multi_head_attention/query/kernel_1,multi_head_attention/attention_output/bias_1dense_3/bias_1!multi_head_attention/query/bias_1layer_normalization/beta_1multi_head_attention/key/bias_1#multi_head_attention/value/kernel_1dense_4/bias_1!multi_head_attention/value/bias_1dense_1/kernel_1dense_5/bias_1dense_2/kernel_1dense_5/kernel_1dense/kernel_1dense/bias_1dense_2/bias_1Const_1*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8� �J *&
f!R
__inference__traced_save_2701
�

StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable!multi_head_attention/key/kernel_1.multi_head_attention/attention_output/kernel_1dense_1/bias_1layer_normalization_1/gamma_1layer_normalization_1/beta_1dense_4/kernel_1layer_normalization/gamma_1dense_3/kernel_1#multi_head_attention/query/kernel_1,multi_head_attention/attention_output/bias_1dense_3/bias_1!multi_head_attention/query/bias_1layer_normalization/beta_1multi_head_attention/key/bias_1#multi_head_attention/value/kernel_1dense_4/bias_1!multi_head_attention/value/bias_1dense_1/kernel_1dense_5/bias_1dense_2/kernel_1dense_5/kernel_1dense/kernel_1dense/bias_1dense_2/bias_1*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8� �J *)
f$R"
 __inference__traced_restore_2866��
��
�/
__inference__traced_save_2701
file_prefix4
"read_disablecopyonread_variable_27: 2
$read_1_disablecopyonread_variable_26: :
$read_2_disablecopyonread_variable_25: 
 6
$read_3_disablecopyonread_variable_24:
 :
$read_4_disablecopyonread_variable_23: 
 6
$read_5_disablecopyonread_variable_22:
 :
$read_6_disablecopyonread_variable_21: 
 6
$read_7_disablecopyonread_variable_20:
 :
$read_8_disablecopyonread_variable_19:
  2
$read_9_disablecopyonread_variable_18: 3
%read_10_disablecopyonread_variable_17:	3
%read_11_disablecopyonread_variable_16: 3
%read_12_disablecopyonread_variable_15: 8
%read_13_disablecopyonread_variable_14:	 �4
%read_14_disablecopyonread_variable_13:	�8
%read_15_disablecopyonread_variable_12:	� 3
%read_16_disablecopyonread_variable_11: 3
%read_17_disablecopyonread_variable_10:	2
$read_18_disablecopyonread_variable_9: 2
$read_19_disablecopyonread_variable_8: 6
$read_20_disablecopyonread_variable_7: @2
$read_21_disablecopyonread_variable_6:@2
$read_22_disablecopyonread_variable_5:	6
$read_23_disablecopyonread_variable_4:@@2
$read_24_disablecopyonread_variable_3:@2
$read_25_disablecopyonread_variable_2:	6
$read_26_disablecopyonread_variable_1:@H0
"read_27_disablecopyonread_variable:HQ
;read_28_disablecopyonread_multi_head_attention_key_kernel_1: 
 ^
Hread_29_disablecopyonread_multi_head_attention_attention_output_kernel_1:
  7
(read_30_disablecopyonread_dense_1_bias_1:	�E
7read_31_disablecopyonread_layer_normalization_1_gamma_1: D
6read_32_disablecopyonread_layer_normalization_1_beta_1: <
*read_33_disablecopyonread_dense_4_kernel_1:@@C
5read_34_disablecopyonread_layer_normalization_gamma_1: <
*read_35_disablecopyonread_dense_3_kernel_1: @S
=read_36_disablecopyonread_multi_head_attention_query_kernel_1: 
 T
Fread_37_disablecopyonread_multi_head_attention_attention_output_bias_1: 6
(read_38_disablecopyonread_dense_3_bias_1:@M
;read_39_disablecopyonread_multi_head_attention_query_bias_1:
 B
4read_40_disablecopyonread_layer_normalization_beta_1: K
9read_41_disablecopyonread_multi_head_attention_key_bias_1:
 S
=read_42_disablecopyonread_multi_head_attention_value_kernel_1: 
 6
(read_43_disablecopyonread_dense_4_bias_1:@M
;read_44_disablecopyonread_multi_head_attention_value_bias_1:
 =
*read_45_disablecopyonread_dense_1_kernel_1:	 �6
(read_46_disablecopyonread_dense_5_bias_1:H=
*read_47_disablecopyonread_dense_2_kernel_1:	� <
*read_48_disablecopyonread_dense_5_kernel_1:@H:
(read_49_disablecopyonread_dense_kernel_1: 4
&read_50_disablecopyonread_dense_bias_1: 6
(read_51_disablecopyonread_dense_2_bias_1: 
savev2_const_1
identity_105��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_variable_27*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_variable_27^Read/DisableCopyOnRead*
_output_shapes

: *
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

: a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

: i
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_variable_26*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_variable_26^Read_1/DisableCopyOnRead*
_output_shapes
: *
dtype0Z

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_2/DisableCopyOnReadDisableCopyOnRead$read_2_disablecopyonread_variable_25*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp$read_2_disablecopyonread_variable_25^Read_2/DisableCopyOnRead*"
_output_shapes
: 
 *
dtype0b

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
 g

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*"
_output_shapes
: 
 i
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_variable_24*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_variable_24^Read_3/DisableCopyOnRead*
_output_shapes

:
 *
dtype0^

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes

:
 c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:
 i
Read_4/DisableCopyOnReadDisableCopyOnRead$read_4_disablecopyonread_variable_23*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp$read_4_disablecopyonread_variable_23^Read_4/DisableCopyOnRead*"
_output_shapes
: 
 *
dtype0b

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
 g

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*"
_output_shapes
: 
 i
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_variable_22*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_variable_22^Read_5/DisableCopyOnRead*
_output_shapes

:
 *
dtype0_
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes

:
 e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:
 i
Read_6/DisableCopyOnReadDisableCopyOnRead$read_6_disablecopyonread_variable_21*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp$read_6_disablecopyonread_variable_21^Read_6/DisableCopyOnRead*"
_output_shapes
: 
 *
dtype0c
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
 i
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*"
_output_shapes
: 
 i
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_variable_20*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_variable_20^Read_7/DisableCopyOnRead*
_output_shapes

:
 *
dtype0_
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes

:
 e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:
 i
Read_8/DisableCopyOnReadDisableCopyOnRead$read_8_disablecopyonread_variable_19*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp$read_8_disablecopyonread_variable_19^Read_8/DisableCopyOnRead*"
_output_shapes
:
  *
dtype0c
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*"
_output_shapes
:
  i
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*"
_output_shapes
:
  i
Read_9/DisableCopyOnReadDisableCopyOnRead$read_9_disablecopyonread_variable_18*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp$read_9_disablecopyonread_variable_18^Read_9/DisableCopyOnRead*
_output_shapes
: *
dtype0[
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: k
Read_10/DisableCopyOnReadDisableCopyOnRead%read_10_disablecopyonread_variable_17*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp%read_10_disablecopyonread_variable_17^Read_10/DisableCopyOnRead*
_output_shapes
:*
dtype0	\
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0	*
_output_shapes
:k
Read_11/DisableCopyOnReadDisableCopyOnRead%read_11_disablecopyonread_variable_16*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp%read_11_disablecopyonread_variable_16^Read_11/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: k
Read_12/DisableCopyOnReadDisableCopyOnRead%read_12_disablecopyonread_variable_15*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp%read_12_disablecopyonread_variable_15^Read_12/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: k
Read_13/DisableCopyOnReadDisableCopyOnRead%read_13_disablecopyonread_variable_14*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp%read_13_disablecopyonread_variable_14^Read_13/DisableCopyOnRead*
_output_shapes
:	 �*
dtype0a
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes
:	 �f
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	 �k
Read_14/DisableCopyOnReadDisableCopyOnRead%read_14_disablecopyonread_variable_13*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp%read_14_disablecopyonread_variable_13^Read_14/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_15/DisableCopyOnReadDisableCopyOnRead%read_15_disablecopyonread_variable_12*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp%read_15_disablecopyonread_variable_12^Read_15/DisableCopyOnRead*
_output_shapes
:	� *
dtype0a
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes
:	� f
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:	� k
Read_16/DisableCopyOnReadDisableCopyOnRead%read_16_disablecopyonread_variable_11*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp%read_16_disablecopyonread_variable_11^Read_16/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: k
Read_17/DisableCopyOnReadDisableCopyOnRead%read_17_disablecopyonread_variable_10*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp%read_17_disablecopyonread_variable_10^Read_17/DisableCopyOnRead*
_output_shapes
:*
dtype0	\
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0	*
_output_shapes
:j
Read_18/DisableCopyOnReadDisableCopyOnRead$read_18_disablecopyonread_variable_9*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp$read_18_disablecopyonread_variable_9^Read_18/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: j
Read_19/DisableCopyOnReadDisableCopyOnRead$read_19_disablecopyonread_variable_8*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp$read_19_disablecopyonread_variable_8^Read_19/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: j
Read_20/DisableCopyOnReadDisableCopyOnRead$read_20_disablecopyonread_variable_7*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp$read_20_disablecopyonread_variable_7^Read_20/DisableCopyOnRead*
_output_shapes

: @*
dtype0`
Identity_40IdentityRead_20/ReadVariableOp:value:0*
T0*
_output_shapes

: @e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

: @j
Read_21/DisableCopyOnReadDisableCopyOnRead$read_21_disablecopyonread_variable_6*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp$read_21_disablecopyonread_variable_6^Read_21/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_42IdentityRead_21/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_22/DisableCopyOnReadDisableCopyOnRead$read_22_disablecopyonread_variable_5*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp$read_22_disablecopyonread_variable_5^Read_22/DisableCopyOnRead*
_output_shapes
:*
dtype0	\
Identity_44IdentityRead_22/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0	*
_output_shapes
:j
Read_23/DisableCopyOnReadDisableCopyOnRead$read_23_disablecopyonread_variable_4*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp$read_23_disablecopyonread_variable_4^Read_23/DisableCopyOnRead*
_output_shapes

:@@*
dtype0`
Identity_46IdentityRead_23/ReadVariableOp:value:0*
T0*
_output_shapes

:@@e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:@@j
Read_24/DisableCopyOnReadDisableCopyOnRead$read_24_disablecopyonread_variable_3*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp$read_24_disablecopyonread_variable_3^Read_24/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_48IdentityRead_24/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_25/DisableCopyOnReadDisableCopyOnRead$read_25_disablecopyonread_variable_2*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp$read_25_disablecopyonread_variable_2^Read_25/DisableCopyOnRead*
_output_shapes
:*
dtype0	\
Identity_50IdentityRead_25/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0	*
_output_shapes
:j
Read_26/DisableCopyOnReadDisableCopyOnRead$read_26_disablecopyonread_variable_1*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp$read_26_disablecopyonread_variable_1^Read_26/DisableCopyOnRead*
_output_shapes

:@H*
dtype0`
Identity_52IdentityRead_26/ReadVariableOp:value:0*
T0*
_output_shapes

:@He
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes

:@Hh
Read_27/DisableCopyOnReadDisableCopyOnRead"read_27_disablecopyonread_variable*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp"read_27_disablecopyonread_variable^Read_27/DisableCopyOnRead*
_output_shapes
:H*
dtype0\
Identity_54IdentityRead_27/ReadVariableOp:value:0*
T0*
_output_shapes
:Ha
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:H�
Read_28/DisableCopyOnReadDisableCopyOnRead;read_28_disablecopyonread_multi_head_attention_key_kernel_1*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp;read_28_disablecopyonread_multi_head_attention_key_kernel_1^Read_28/DisableCopyOnRead*"
_output_shapes
: 
 *
dtype0d
Identity_56IdentityRead_28/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
 i
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*"
_output_shapes
: 
 �
Read_29/DisableCopyOnReadDisableCopyOnReadHread_29_disablecopyonread_multi_head_attention_attention_output_kernel_1*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpHread_29_disablecopyonread_multi_head_attention_attention_output_kernel_1^Read_29/DisableCopyOnRead*"
_output_shapes
:
  *
dtype0d
Identity_58IdentityRead_29/ReadVariableOp:value:0*
T0*"
_output_shapes
:
  i
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*"
_output_shapes
:
  n
Read_30/DisableCopyOnReadDisableCopyOnRead(read_30_disablecopyonread_dense_1_bias_1*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp(read_30_disablecopyonread_dense_1_bias_1^Read_30/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_60IdentityRead_30/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_31/DisableCopyOnReadDisableCopyOnRead7read_31_disablecopyonread_layer_normalization_1_gamma_1*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp7read_31_disablecopyonread_layer_normalization_1_gamma_1^Read_31/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_62IdentityRead_31/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_32/DisableCopyOnReadDisableCopyOnRead6read_32_disablecopyonread_layer_normalization_1_beta_1*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp6read_32_disablecopyonread_layer_normalization_1_beta_1^Read_32/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_64IdentityRead_32/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: p
Read_33/DisableCopyOnReadDisableCopyOnRead*read_33_disablecopyonread_dense_4_kernel_1*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp*read_33_disablecopyonread_dense_4_kernel_1^Read_33/DisableCopyOnRead*
_output_shapes

:@@*
dtype0`
Identity_66IdentityRead_33/ReadVariableOp:value:0*
T0*
_output_shapes

:@@e
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes

:@@{
Read_34/DisableCopyOnReadDisableCopyOnRead5read_34_disablecopyonread_layer_normalization_gamma_1*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp5read_34_disablecopyonread_layer_normalization_gamma_1^Read_34/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_68IdentityRead_34/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
: p
Read_35/DisableCopyOnReadDisableCopyOnRead*read_35_disablecopyonread_dense_3_kernel_1*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp*read_35_disablecopyonread_dense_3_kernel_1^Read_35/DisableCopyOnRead*
_output_shapes

: @*
dtype0`
Identity_70IdentityRead_35/ReadVariableOp:value:0*
T0*
_output_shapes

: @e
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes

: @�
Read_36/DisableCopyOnReadDisableCopyOnRead=read_36_disablecopyonread_multi_head_attention_query_kernel_1*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp=read_36_disablecopyonread_multi_head_attention_query_kernel_1^Read_36/DisableCopyOnRead*"
_output_shapes
: 
 *
dtype0d
Identity_72IdentityRead_36/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
 i
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*"
_output_shapes
: 
 �
Read_37/DisableCopyOnReadDisableCopyOnReadFread_37_disablecopyonread_multi_head_attention_attention_output_bias_1*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOpFread_37_disablecopyonread_multi_head_attention_attention_output_bias_1^Read_37/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_74IdentityRead_37/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
: n
Read_38/DisableCopyOnReadDisableCopyOnRead(read_38_disablecopyonread_dense_3_bias_1*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp(read_38_disablecopyonread_dense_3_bias_1^Read_38/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_76IdentityRead_38/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_39/DisableCopyOnReadDisableCopyOnRead;read_39_disablecopyonread_multi_head_attention_query_bias_1*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp;read_39_disablecopyonread_multi_head_attention_query_bias_1^Read_39/DisableCopyOnRead*
_output_shapes

:
 *
dtype0`
Identity_78IdentityRead_39/ReadVariableOp:value:0*
T0*
_output_shapes

:
 e
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes

:
 z
Read_40/DisableCopyOnReadDisableCopyOnRead4read_40_disablecopyonread_layer_normalization_beta_1*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp4read_40_disablecopyonread_layer_normalization_beta_1^Read_40/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_80IdentityRead_40/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_41/DisableCopyOnReadDisableCopyOnRead9read_41_disablecopyonread_multi_head_attention_key_bias_1*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp9read_41_disablecopyonread_multi_head_attention_key_bias_1^Read_41/DisableCopyOnRead*
_output_shapes

:
 *
dtype0`
Identity_82IdentityRead_41/ReadVariableOp:value:0*
T0*
_output_shapes

:
 e
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes

:
 �
Read_42/DisableCopyOnReadDisableCopyOnRead=read_42_disablecopyonread_multi_head_attention_value_kernel_1*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp=read_42_disablecopyonread_multi_head_attention_value_kernel_1^Read_42/DisableCopyOnRead*"
_output_shapes
: 
 *
dtype0d
Identity_84IdentityRead_42/ReadVariableOp:value:0*
T0*"
_output_shapes
: 
 i
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*"
_output_shapes
: 
 n
Read_43/DisableCopyOnReadDisableCopyOnRead(read_43_disablecopyonread_dense_4_bias_1*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp(read_43_disablecopyonread_dense_4_bias_1^Read_43/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_86IdentityRead_43/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_44/DisableCopyOnReadDisableCopyOnRead;read_44_disablecopyonread_multi_head_attention_value_bias_1*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp;read_44_disablecopyonread_multi_head_attention_value_bias_1^Read_44/DisableCopyOnRead*
_output_shapes

:
 *
dtype0`
Identity_88IdentityRead_44/ReadVariableOp:value:0*
T0*
_output_shapes

:
 e
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes

:
 p
Read_45/DisableCopyOnReadDisableCopyOnRead*read_45_disablecopyonread_dense_1_kernel_1*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp*read_45_disablecopyonread_dense_1_kernel_1^Read_45/DisableCopyOnRead*
_output_shapes
:	 �*
dtype0a
Identity_90IdentityRead_45/ReadVariableOp:value:0*
T0*
_output_shapes
:	 �f
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:	 �n
Read_46/DisableCopyOnReadDisableCopyOnRead(read_46_disablecopyonread_dense_5_bias_1*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp(read_46_disablecopyonread_dense_5_bias_1^Read_46/DisableCopyOnRead*
_output_shapes
:H*
dtype0\
Identity_92IdentityRead_46/ReadVariableOp:value:0*
T0*
_output_shapes
:Ha
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:Hp
Read_47/DisableCopyOnReadDisableCopyOnRead*read_47_disablecopyonread_dense_2_kernel_1*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp*read_47_disablecopyonread_dense_2_kernel_1^Read_47/DisableCopyOnRead*
_output_shapes
:	� *
dtype0a
Identity_94IdentityRead_47/ReadVariableOp:value:0*
T0*
_output_shapes
:	� f
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:	� p
Read_48/DisableCopyOnReadDisableCopyOnRead*read_48_disablecopyonread_dense_5_kernel_1*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp*read_48_disablecopyonread_dense_5_kernel_1^Read_48/DisableCopyOnRead*
_output_shapes

:@H*
dtype0`
Identity_96IdentityRead_48/ReadVariableOp:value:0*
T0*
_output_shapes

:@He
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes

:@Hn
Read_49/DisableCopyOnReadDisableCopyOnRead(read_49_disablecopyonread_dense_kernel_1*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp(read_49_disablecopyonread_dense_kernel_1^Read_49/DisableCopyOnRead*
_output_shapes

: *
dtype0`
Identity_98IdentityRead_49/ReadVariableOp:value:0*
T0*
_output_shapes

: e
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes

: l
Read_50/DisableCopyOnReadDisableCopyOnRead&read_50_disablecopyonread_dense_bias_1*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp&read_50_disablecopyonread_dense_bias_1^Read_50/DisableCopyOnRead*
_output_shapes
: *
dtype0]
Identity_100IdentityRead_50/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
: n
Read_51/DisableCopyOnReadDisableCopyOnRead(read_51_disablecopyonread_dense_2_bias_1*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp(read_51_disablecopyonread_dense_2_bias_1^Read_51/DisableCopyOnRead*
_output_shapes
: *
dtype0]
Identity_102IdentityRead_51/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*�
value�B�5B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0savev2_const_1"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *C
dtypes9
725				�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_104Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_105IdentityIdentity_104:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_105Identity_105:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:?5;

_output_shapes
: 
!
_user_specified_name	Const_1:.4*
(
_user_specified_namedense_2/bias_1:,3(
&
_user_specified_namedense/bias_1:.2*
(
_user_specified_namedense/kernel_1:01,
*
_user_specified_namedense_5/kernel_1:00,
*
_user_specified_namedense_2/kernel_1:./*
(
_user_specified_namedense_5/bias_1:0.,
*
_user_specified_namedense_1/kernel_1:A-=
;
_user_specified_name#!multi_head_attention/value/bias_1:.,*
(
_user_specified_namedense_4/bias_1:C+?
=
_user_specified_name%#multi_head_attention/value/kernel_1:?*;
9
_user_specified_name!multi_head_attention/key/bias_1::)6
4
_user_specified_namelayer_normalization/beta_1:A(=
;
_user_specified_name#!multi_head_attention/query/bias_1:.'*
(
_user_specified_namedense_3/bias_1:L&H
F
_user_specified_name.,multi_head_attention/attention_output/bias_1:C%?
=
_user_specified_name%#multi_head_attention/query/kernel_1:0$,
*
_user_specified_namedense_3/kernel_1:;#7
5
_user_specified_namelayer_normalization/gamma_1:0",
*
_user_specified_namedense_4/kernel_1:<!8
6
_user_specified_namelayer_normalization_1/beta_1:= 9
7
_user_specified_namelayer_normalization_1/gamma_1:.*
(
_user_specified_namedense_1/bias_1:NJ
H
_user_specified_name0.multi_head_attention/attention_output/kernel_1:A=
;
_user_specified_name#!multi_head_attention/key/kernel_1:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_17:+
'
%
_user_specified_nameVariable_18:+	'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_23:+'
%
_user_specified_nameVariable_24:+'
%
_user_specified_nameVariable_25:+'
%
_user_specified_nameVariable_26:+'
%
_user_specified_nameVariable_27:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
!__inference_internal_grad_fn_2474
result_grads_0
result_grads_1
result_grads_2#
mul_functional_1_dense_4_1_beta&
"mul_functional_1_dense_4_1_biasadd
identity

identity_1�
mulMulmul_functional_1_dense_4_1_beta"mul_functional_1_dense_4_1_biasadd^result_grads_0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@�
mul_1Mulmul_functional_1_dense_4_1_beta"mul_functional_1_dense_4_1_biasadd*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������@f
SquareSquare"mul_functional_1_dense_4_1_biasadd*
T0*'
_output_shapes
:���������@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:���������@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:���������@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:���������@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:���������@:���������@: : :���������@:gc
'
_output_shapes
:���������@
8
_user_specified_name functional_1/dense_4_1/BiasAdd:SO

_output_shapes
: 
5
_user_specified_namefunctional_1/dense_4_1/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0
�
�
+__inference_signature_wrapper___call___2198
keras_tensor
unknown: 
	unknown_0: 
	unknown_1
	unknown_2: 
 
	unknown_3:
 
	unknown_4: 
 
	unknown_5:
 
	unknown_6: 
 
	unknown_7:
 
	unknown_8:
  
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:	 �

unknown_13:	�

unknown_14:	� 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: @

unknown_19:@

unknown_20:@@

unknown_21:@

unknown_22:@H

unknown_23:H
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallkeras_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*:
_read_only_resource_inputs
	
*5
config_proto%#

CPU

GPU2*0J 8� �J *"
fR
__inference___call___2087s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:������������������: : :� : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2194:$ 

_user_specified_name2192:$ 

_user_specified_name2190:$ 

_user_specified_name2188:$ 

_user_specified_name2186:$ 

_user_specified_name2184:$ 

_user_specified_name2182:$ 

_user_specified_name2180:$ 

_user_specified_name2178:$ 

_user_specified_name2176:$ 

_user_specified_name2174:$ 

_user_specified_name2172:$ 

_user_specified_name2170:$ 

_user_specified_name2168:$ 

_user_specified_name2166:$
 

_user_specified_name2164:$	 

_user_specified_name2162:$ 

_user_specified_name2160:$ 

_user_specified_name2158:$ 

_user_specified_name2156:$ 

_user_specified_name2154:$ 

_user_specified_name2152:IE
#
_output_shapes
:� 

_user_specified_name2150:$ 

_user_specified_name2148:$ 

_user_specified_name2146:b ^
4
_output_shapes"
 :������������������
&
_user_specified_namekeras_tensor
�
�
!__inference_internal_grad_fn_2447
result_grads_0
result_grads_1
result_grads_2#
mul_functional_1_dense_3_1_beta&
"mul_functional_1_dense_3_1_biasadd
identity

identity_1�
mulMulmul_functional_1_dense_3_1_beta"mul_functional_1_dense_3_1_biasadd^result_grads_0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@�
mul_1Mulmul_functional_1_dense_3_1_beta"mul_functional_1_dense_3_1_biasadd*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������@f
SquareSquare"mul_functional_1_dense_3_1_biasadd*
T0*'
_output_shapes
:���������@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:���������@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:���������@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:���������@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:���������@:���������@: : :���������@:gc
'
_output_shapes
:���������@
8
_user_specified_name functional_1/dense_3_1/BiasAdd:SO

_output_shapes
: 
5
_user_specified_namefunctional_1/dense_3_1/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0
��
�
 __inference__traced_restore_2866
file_prefix.
assignvariableop_variable_27: ,
assignvariableop_1_variable_26: 4
assignvariableop_2_variable_25: 
 0
assignvariableop_3_variable_24:
 4
assignvariableop_4_variable_23: 
 0
assignvariableop_5_variable_22:
 4
assignvariableop_6_variable_21: 
 0
assignvariableop_7_variable_20:
 4
assignvariableop_8_variable_19:
  ,
assignvariableop_9_variable_18: -
assignvariableop_10_variable_17:	-
assignvariableop_11_variable_16: -
assignvariableop_12_variable_15: 2
assignvariableop_13_variable_14:	 �.
assignvariableop_14_variable_13:	�2
assignvariableop_15_variable_12:	� -
assignvariableop_16_variable_11: -
assignvariableop_17_variable_10:	,
assignvariableop_18_variable_9: ,
assignvariableop_19_variable_8: 0
assignvariableop_20_variable_7: @,
assignvariableop_21_variable_6:@,
assignvariableop_22_variable_5:	0
assignvariableop_23_variable_4:@@,
assignvariableop_24_variable_3:@,
assignvariableop_25_variable_2:	0
assignvariableop_26_variable_1:@H*
assignvariableop_27_variable:HK
5assignvariableop_28_multi_head_attention_key_kernel_1: 
 X
Bassignvariableop_29_multi_head_attention_attention_output_kernel_1:
  1
"assignvariableop_30_dense_1_bias_1:	�?
1assignvariableop_31_layer_normalization_1_gamma_1: >
0assignvariableop_32_layer_normalization_1_beta_1: 6
$assignvariableop_33_dense_4_kernel_1:@@=
/assignvariableop_34_layer_normalization_gamma_1: 6
$assignvariableop_35_dense_3_kernel_1: @M
7assignvariableop_36_multi_head_attention_query_kernel_1: 
 N
@assignvariableop_37_multi_head_attention_attention_output_bias_1: 0
"assignvariableop_38_dense_3_bias_1:@G
5assignvariableop_39_multi_head_attention_query_bias_1:
 <
.assignvariableop_40_layer_normalization_beta_1: E
3assignvariableop_41_multi_head_attention_key_bias_1:
 M
7assignvariableop_42_multi_head_attention_value_kernel_1: 
 0
"assignvariableop_43_dense_4_bias_1:@G
5assignvariableop_44_multi_head_attention_value_bias_1:
 7
$assignvariableop_45_dense_1_kernel_1:	 �0
"assignvariableop_46_dense_5_bias_1:H7
$assignvariableop_47_dense_2_kernel_1:	� 6
$assignvariableop_48_dense_5_kernel_1:@H4
"assignvariableop_49_dense_kernel_1: .
 assignvariableop_50_dense_bias_1: 0
"assignvariableop_51_dense_2_bias_1: 
identity_53��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*�
value�B�5B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725				[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_27Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_26Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_25Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_24Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_23Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_22Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_21Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_20Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_19Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_18Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_17Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_16Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_15Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variable_14Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_13Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_12Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_11Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_10Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_9Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variable_8Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_7Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_6Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_5Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_variable_4Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_variable_3Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_variable_2Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_variable_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_variableIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp5assignvariableop_28_multi_head_attention_key_kernel_1Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpBassignvariableop_29_multi_head_attention_attention_output_kernel_1Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp"assignvariableop_30_dense_1_bias_1Identity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp1assignvariableop_31_layer_normalization_1_gamma_1Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp0assignvariableop_32_layer_normalization_1_beta_1Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp$assignvariableop_33_dense_4_kernel_1Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp/assignvariableop_34_layer_normalization_gamma_1Identity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp$assignvariableop_35_dense_3_kernel_1Identity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp7assignvariableop_36_multi_head_attention_query_kernel_1Identity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp@assignvariableop_37_multi_head_attention_attention_output_bias_1Identity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp"assignvariableop_38_dense_3_bias_1Identity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp5assignvariableop_39_multi_head_attention_query_bias_1Identity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp.assignvariableop_40_layer_normalization_beta_1Identity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp3assignvariableop_41_multi_head_attention_key_bias_1Identity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp7assignvariableop_42_multi_head_attention_value_kernel_1Identity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp"assignvariableop_43_dense_4_bias_1Identity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp5assignvariableop_44_multi_head_attention_value_bias_1Identity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp$assignvariableop_45_dense_1_kernel_1Identity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_5_bias_1Identity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp$assignvariableop_47_dense_2_kernel_1Identity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp$assignvariableop_48_dense_5_kernel_1Identity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp"assignvariableop_49_dense_kernel_1Identity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp assignvariableop_50_dense_bias_1Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp"assignvariableop_51_dense_2_bias_1Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �	
Identity_52Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_53IdentityIdentity_52:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_53Identity_53:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:.4*
(
_user_specified_namedense_2/bias_1:,3(
&
_user_specified_namedense/bias_1:.2*
(
_user_specified_namedense/kernel_1:01,
*
_user_specified_namedense_5/kernel_1:00,
*
_user_specified_namedense_2/kernel_1:./*
(
_user_specified_namedense_5/bias_1:0.,
*
_user_specified_namedense_1/kernel_1:A-=
;
_user_specified_name#!multi_head_attention/value/bias_1:.,*
(
_user_specified_namedense_4/bias_1:C+?
=
_user_specified_name%#multi_head_attention/value/kernel_1:?*;
9
_user_specified_name!multi_head_attention/key/bias_1::)6
4
_user_specified_namelayer_normalization/beta_1:A(=
;
_user_specified_name#!multi_head_attention/query/bias_1:.'*
(
_user_specified_namedense_3/bias_1:L&H
F
_user_specified_name.,multi_head_attention/attention_output/bias_1:C%?
=
_user_specified_name%#multi_head_attention/query/kernel_1:0$,
*
_user_specified_namedense_3/kernel_1:;#7
5
_user_specified_namelayer_normalization/gamma_1:0",
*
_user_specified_namedense_4/kernel_1:<!8
6
_user_specified_namelayer_normalization_1/beta_1:= 9
7
_user_specified_namelayer_normalization_1/gamma_1:.*
(
_user_specified_namedense_1/bias_1:NJ
H
_user_specified_name0.multi_head_attention/attention_output/kernel_1:A=
;
_user_specified_name#!multi_head_attention/key/kernel_1:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_17:+
'
%
_user_specified_nameVariable_18:+	'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_23:+'
%
_user_specified_nameVariable_24:+'
%
_user_specified_nameVariable_25:+'
%
_user_specified_nameVariable_26:+'
%
_user_specified_nameVariable_27:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
+__inference_signature_wrapper___call___2143
keras_tensor
unknown: 
	unknown_0: 
	unknown_1
	unknown_2: 
 
	unknown_3:
 
	unknown_4: 
 
	unknown_5:
 
	unknown_6: 
 
	unknown_7:
 
	unknown_8:
  
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:	 �

unknown_13:	�

unknown_14:	� 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: @

unknown_19:@

unknown_20:@@

unknown_21:@

unknown_22:@H

unknown_23:H
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallkeras_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*:
_read_only_resource_inputs
	
*5
config_proto%#

CPU

GPU2*0J 8� �J *"
fR
__inference___call___2087s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:������������������: : :� : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2139:$ 

_user_specified_name2137:$ 

_user_specified_name2135:$ 

_user_specified_name2133:$ 

_user_specified_name2131:$ 

_user_specified_name2129:$ 

_user_specified_name2127:$ 

_user_specified_name2125:$ 

_user_specified_name2123:$ 

_user_specified_name2121:$ 

_user_specified_name2119:$ 

_user_specified_name2117:$ 

_user_specified_name2115:$ 

_user_specified_name2113:$ 

_user_specified_name2111:$
 

_user_specified_name2109:$	 

_user_specified_name2107:$ 

_user_specified_name2105:$ 

_user_specified_name2103:$ 

_user_specified_name2101:$ 

_user_specified_name2099:$ 

_user_specified_name2097:IE
#
_output_shapes
:� 

_user_specified_name2095:$ 

_user_specified_name2093:$ 

_user_specified_name2091:b ^
4
_output_shapes"
 :������������������
&
_user_specified_namekeras_tensor
��
�
__inference___call___2087
keras_tensorC
1functional_1_dense_1_cast_readvariableop_resource: B
4functional_1_dense_1_biasadd_readvariableop_resource: 1
-functional_1_positional_encoding_layer_1_1929^
Hfunctional_1_multi_head_attention_1_query_1_cast_readvariableop_resource: 
 Y
Gfunctional_1_multi_head_attention_1_query_1_add_readvariableop_resource:
 \
Ffunctional_1_multi_head_attention_1_key_1_cast_readvariableop_resource: 
 W
Efunctional_1_multi_head_attention_1_key_1_add_readvariableop_resource:
 ^
Hfunctional_1_multi_head_attention_1_value_1_cast_readvariableop_resource: 
 Y
Gfunctional_1_multi_head_attention_1_value_1_add_readvariableop_resource:
 i
Sfunctional_1_multi_head_attention_1_attention_output_1_cast_readvariableop_resource:
  `
Rfunctional_1_multi_head_attention_1_attention_output_1_add_readvariableop_resource: P
Bfunctional_1_layer_normalization_1_reshape_readvariableop_resource: R
Dfunctional_1_layer_normalization_1_reshape_1_readvariableop_resource: F
3functional_1_dense_1_2_cast_readvariableop_resource:	 �E
6functional_1_dense_1_2_biasadd_readvariableop_resource:	�F
3functional_1_dense_2_1_cast_readvariableop_resource:	� D
6functional_1_dense_2_1_biasadd_readvariableop_resource: R
Dfunctional_1_layer_normalization_1_2_reshape_readvariableop_resource: T
Ffunctional_1_layer_normalization_1_2_reshape_1_readvariableop_resource: E
3functional_1_dense_3_1_cast_readvariableop_resource: @D
6functional_1_dense_3_1_biasadd_readvariableop_resource:@E
3functional_1_dense_4_1_cast_readvariableop_resource:@@D
6functional_1_dense_4_1_biasadd_readvariableop_resource:@E
3functional_1_dense_5_1_cast_readvariableop_resource:@HD
6functional_1_dense_5_1_biasadd_readvariableop_resource:H
identity��+functional_1/dense_1/BiasAdd/ReadVariableOp�(functional_1/dense_1/Cast/ReadVariableOp�-functional_1/dense_1_2/BiasAdd/ReadVariableOp�*functional_1/dense_1_2/Cast/ReadVariableOp�-functional_1/dense_2_1/BiasAdd/ReadVariableOp�*functional_1/dense_2_1/Cast/ReadVariableOp�-functional_1/dense_3_1/BiasAdd/ReadVariableOp�*functional_1/dense_3_1/Cast/ReadVariableOp�-functional_1/dense_4_1/BiasAdd/ReadVariableOp�*functional_1/dense_4_1/Cast/ReadVariableOp�-functional_1/dense_5_1/BiasAdd/ReadVariableOp�*functional_1/dense_5_1/Cast/ReadVariableOp�9functional_1/layer_normalization_1/Reshape/ReadVariableOp�;functional_1/layer_normalization_1/Reshape_1/ReadVariableOp�;functional_1/layer_normalization_1_2/Reshape/ReadVariableOp�=functional_1/layer_normalization_1_2/Reshape_1/ReadVariableOp�Jfunctional_1/multi_head_attention_1/attention_output_1/Cast/ReadVariableOp�Ifunctional_1/multi_head_attention_1/attention_output_1/add/ReadVariableOp�=functional_1/multi_head_attention_1/key_1/Cast/ReadVariableOp�<functional_1/multi_head_attention_1/key_1/add/ReadVariableOp�?functional_1/multi_head_attention_1/query_1/Cast/ReadVariableOp�>functional_1/multi_head_attention_1/query_1/add/ReadVariableOp�?functional_1/multi_head_attention_1/value_1/Cast/ReadVariableOp�>functional_1/multi_head_attention_1/value_1/add/ReadVariableOp�
(functional_1/dense_1/Cast/ReadVariableOpReadVariableOp1functional_1_dense_1_cast_readvariableop_resource*
_output_shapes

: *
dtype0�
functional_1/dense_1/MatMulBatchMatMulV2keras_tensor0functional_1/dense_1/Cast/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������ �
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
functional_1/dense_1/BiasAddBiasAdd$functional_1/dense_1/MatMul:output:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������ �
.functional_1/positional_encoding_layer_1/ShapeShape%functional_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
::���
<functional_1/positional_encoding_layer_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:�
>functional_1/positional_encoding_layer_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
>functional_1/positional_encoding_layer_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
6functional_1/positional_encoding_layer_1/strided_sliceStridedSlice7functional_1/positional_encoding_layer_1/Shape:output:0Efunctional_1/positional_encoding_layer_1/strided_slice/stack:output:0Gfunctional_1/positional_encoding_layer_1/strided_slice/stack_1:output:0Gfunctional_1/positional_encoding_layer_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.functional_1/positional_encoding_layer_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
0functional_1/positional_encoding_layer_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :�
@functional_1/positional_encoding_layer_1/strided_slice_1/stack/0Const*
_output_shapes
: *
dtype0*
value	B : �
@functional_1/positional_encoding_layer_1/strided_slice_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : �
>functional_1/positional_encoding_layer_1/strided_slice_1/stackPackIfunctional_1/positional_encoding_layer_1/strided_slice_1/stack/0:output:07functional_1/positional_encoding_layer_1/Const:output:0Ifunctional_1/positional_encoding_layer_1/strided_slice_1/stack/2:output:0*
N*
T0*
_output_shapes
:�
Bfunctional_1/positional_encoding_layer_1/strided_slice_1/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : �
Bfunctional_1/positional_encoding_layer_1/strided_slice_1/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : �
@functional_1/positional_encoding_layer_1/strided_slice_1/stack_1PackKfunctional_1/positional_encoding_layer_1/strided_slice_1/stack_1/0:output:0?functional_1/positional_encoding_layer_1/strided_slice:output:0Kfunctional_1/positional_encoding_layer_1/strided_slice_1/stack_1/2:output:0*
N*
T0*
_output_shapes
:�
Bfunctional_1/positional_encoding_layer_1/strided_slice_1/stack_2/0Const*
_output_shapes
: *
dtype0*
value	B :�
Bfunctional_1/positional_encoding_layer_1/strided_slice_1/stack_2/2Const*
_output_shapes
: *
dtype0*
value	B :�
@functional_1/positional_encoding_layer_1/strided_slice_1/stack_2PackKfunctional_1/positional_encoding_layer_1/strided_slice_1/stack_2/0:output:09functional_1/positional_encoding_layer_1/Const_1:output:0Kfunctional_1/positional_encoding_layer_1/strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:�
8functional_1/positional_encoding_layer_1/strided_slice_1StridedSlice-functional_1_positional_encoding_layer_1_1929Gfunctional_1/positional_encoding_layer_1/strided_slice_1/stack:output:0Ifunctional_1/positional_encoding_layer_1/strided_slice_1/stack_1:output:0Ifunctional_1/positional_encoding_layer_1/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:��������� *

begin_mask*
end_mask�
,functional_1/positional_encoding_layer_1/addAddV2%functional_1/dense_1/BiasAdd:output:0Afunctional_1/positional_encoding_layer_1/strided_slice_1:output:0*
T0*4
_output_shapes"
 :������������������ �
?functional_1/multi_head_attention_1/query_1/Cast/ReadVariableOpReadVariableOpHfunctional_1_multi_head_attention_1_query_1_cast_readvariableop_resource*"
_output_shapes
: 
 *
dtype0�
9functional_1/multi_head_attention_1/query_1/einsum/EinsumEinsum0functional_1/positional_encoding_layer_1/add:z:0Gfunctional_1/multi_head_attention_1/query_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������
 *
equationabc,cde->abde�
>functional_1/multi_head_attention_1/query_1/add/ReadVariableOpReadVariableOpGfunctional_1_multi_head_attention_1_query_1_add_readvariableop_resource*
_output_shapes

:
 *
dtype0�
/functional_1/multi_head_attention_1/query_1/addAddV2Bfunctional_1/multi_head_attention_1/query_1/einsum/Einsum:output:0Ffunctional_1/multi_head_attention_1/query_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������
 �
=functional_1/multi_head_attention_1/key_1/Cast/ReadVariableOpReadVariableOpFfunctional_1_multi_head_attention_1_key_1_cast_readvariableop_resource*"
_output_shapes
: 
 *
dtype0�
7functional_1/multi_head_attention_1/key_1/einsum/EinsumEinsum0functional_1/positional_encoding_layer_1/add:z:0Efunctional_1/multi_head_attention_1/key_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������
 *
equationabc,cde->abde�
<functional_1/multi_head_attention_1/key_1/add/ReadVariableOpReadVariableOpEfunctional_1_multi_head_attention_1_key_1_add_readvariableop_resource*
_output_shapes

:
 *
dtype0�
-functional_1/multi_head_attention_1/key_1/addAddV2@functional_1/multi_head_attention_1/key_1/einsum/Einsum:output:0Dfunctional_1/multi_head_attention_1/key_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������
 �
?functional_1/multi_head_attention_1/value_1/Cast/ReadVariableOpReadVariableOpHfunctional_1_multi_head_attention_1_value_1_cast_readvariableop_resource*"
_output_shapes
: 
 *
dtype0�
9functional_1/multi_head_attention_1/value_1/einsum/EinsumEinsum0functional_1/positional_encoding_layer_1/add:z:0Gfunctional_1/multi_head_attention_1/value_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������
 *
equationabc,cde->abde�
>functional_1/multi_head_attention_1/value_1/add/ReadVariableOpReadVariableOpGfunctional_1_multi_head_attention_1_value_1_add_readvariableop_resource*
_output_shapes

:
 *
dtype0�
/functional_1/multi_head_attention_1/value_1/addAddV2Bfunctional_1/multi_head_attention_1/value_1/einsum/Einsum:output:0Ffunctional_1/multi_head_attention_1/value_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������
 �
)functional_1/multi_head_attention_1/ShapeShape1functional_1/multi_head_attention_1/key_1/add:z:0*
T0*
_output_shapes
::���
7functional_1/multi_head_attention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
9functional_1/multi_head_attention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
9functional_1/multi_head_attention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1functional_1/multi_head_attention_1/strided_sliceStridedSlice2functional_1/multi_head_attention_1/Shape:output:0@functional_1/multi_head_attention_1/strided_slice/stack:output:0Bfunctional_1/multi_head_attention_1/strided_slice/stack_1:output:0Bfunctional_1/multi_head_attention_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
1functional_1/multi_head_attention_1/einsum/EinsumEinsum3functional_1/multi_head_attention_1/query_1/add:z:01functional_1/multi_head_attention_1/key_1/add:z:0*
N*
T0*A
_output_shapes/
-:+���������
������������������*
equationBTNH,BSNH->BNTSo
*functional_1/multi_head_attention_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *�5>�
'functional_1/multi_head_attention_1/MulMul:functional_1/multi_head_attention_1/einsum/Einsum:output:03functional_1/multi_head_attention_1/Cast/x:output:0*
T0*A
_output_shapes/
-:+���������
�������������������
+functional_1/multi_head_attention_1/SoftmaxSoftmax+functional_1/multi_head_attention_1/Mul:z:0*
T0*A
_output_shapes/
-:+���������
�������������������
3functional_1/multi_head_attention_1/einsum_1/EinsumEinsum5functional_1/multi_head_attention_1/Softmax:softmax:03functional_1/multi_head_attention_1/value_1/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������
 *
equationBNTS,BSNH->BTNH�
Jfunctional_1/multi_head_attention_1/attention_output_1/Cast/ReadVariableOpReadVariableOpSfunctional_1_multi_head_attention_1_attention_output_1_cast_readvariableop_resource*"
_output_shapes
:
  *
dtype0�
Dfunctional_1/multi_head_attention_1/attention_output_1/einsum/EinsumEinsum<functional_1/multi_head_attention_1/einsum_1/Einsum:output:0Rfunctional_1/multi_head_attention_1/attention_output_1/Cast/ReadVariableOp:value:0*
N*
T0*4
_output_shapes"
 :������������������ *
equationabcd,cde->abe�
Ifunctional_1/multi_head_attention_1/attention_output_1/add/ReadVariableOpReadVariableOpRfunctional_1_multi_head_attention_1_attention_output_1_add_readvariableop_resource*
_output_shapes
: *
dtype0�
:functional_1/multi_head_attention_1/attention_output_1/addAddV2Mfunctional_1/multi_head_attention_1/attention_output_1/einsum/Einsum:output:0Qfunctional_1/multi_head_attention_1/attention_output_1/add/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������ �
functional_1/add_1/AddAddV20functional_1/positional_encoding_layer_1/add:z:0>functional_1/multi_head_attention_1/attention_output_1/add:z:0*
T0*4
_output_shapes"
 :������������������ �
Afunctional_1/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
/functional_1/layer_normalization_1/moments/meanMeanfunctional_1/add_1/Add:z:0Jfunctional_1/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
7functional_1/layer_normalization_1/moments/StopGradientStopGradient8functional_1/layer_normalization_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
<functional_1/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencefunctional_1/add_1/Add:z:0@functional_1/layer_normalization_1/moments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ �
Efunctional_1/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
3functional_1/layer_normalization_1/moments/varianceMean@functional_1/layer_normalization_1/moments/SquaredDifference:z:0Nfunctional_1/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
9functional_1/layer_normalization_1/Reshape/ReadVariableOpReadVariableOpBfunctional_1_layer_normalization_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0�
0functional_1/layer_normalization_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"          �
*functional_1/layer_normalization_1/ReshapeReshapeAfunctional_1/layer_normalization_1/Reshape/ReadVariableOp:value:09functional_1/layer_normalization_1/Reshape/shape:output:0*
T0*"
_output_shapes
: �
;functional_1/layer_normalization_1/Reshape_1/ReadVariableOpReadVariableOpDfunctional_1_layer_normalization_1_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype0�
2functional_1/layer_normalization_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"          �
,functional_1/layer_normalization_1/Reshape_1ReshapeCfunctional_1/layer_normalization_1/Reshape_1/ReadVariableOp:value:0;functional_1/layer_normalization_1/Reshape_1/shape:output:0*
T0*"
_output_shapes
: m
(functional_1/layer_normalization_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
&functional_1/layer_normalization_1/addAddV2<functional_1/layer_normalization_1/moments/variance:output:01functional_1/layer_normalization_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
(functional_1/layer_normalization_1/RsqrtRsqrt*functional_1/layer_normalization_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
&functional_1/layer_normalization_1/mulMul,functional_1/layer_normalization_1/Rsqrt:y:03functional_1/layer_normalization_1/Reshape:output:0*
T0*4
_output_shapes"
 :������������������ �
&functional_1/layer_normalization_1/NegNeg8functional_1/layer_normalization_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
(functional_1/layer_normalization_1/mul_1Mul*functional_1/layer_normalization_1/Neg:y:0*functional_1/layer_normalization_1/mul:z:0*
T0*4
_output_shapes"
 :������������������ �
(functional_1/layer_normalization_1/add_1AddV2,functional_1/layer_normalization_1/mul_1:z:05functional_1/layer_normalization_1/Reshape_1:output:0*
T0*4
_output_shapes"
 :������������������ �
(functional_1/layer_normalization_1/mul_2Mulfunctional_1/add_1/Add:z:0*functional_1/layer_normalization_1/mul:z:0*
T0*4
_output_shapes"
 :������������������ �
(functional_1/layer_normalization_1/add_2AddV2,functional_1/layer_normalization_1/mul_2:z:0,functional_1/layer_normalization_1/add_1:z:0*
T0*4
_output_shapes"
 :������������������ �
*functional_1/dense_1_2/Cast/ReadVariableOpReadVariableOp3functional_1_dense_1_2_cast_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
functional_1/dense_1_2/MatMulBatchMatMulV2,functional_1/layer_normalization_1/add_2:z:02functional_1/dense_1_2/Cast/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
-functional_1/dense_1_2/BiasAdd/ReadVariableOpReadVariableOp6functional_1_dense_1_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
functional_1/dense_1_2/BiasAddBiasAdd&functional_1/dense_1_2/MatMul:output:05functional_1/dense_1_2/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
functional_1/dense_1_2/ReluRelu'functional_1/dense_1_2/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
*functional_1/dense_2_1/Cast/ReadVariableOpReadVariableOp3functional_1_dense_2_1_cast_readvariableop_resource*
_output_shapes
:	� *
dtype0�
functional_1/dense_2_1/MatMulBatchMatMulV2)functional_1/dense_1_2/Relu:activations:02functional_1/dense_2_1/Cast/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������ �
-functional_1/dense_2_1/BiasAdd/ReadVariableOpReadVariableOp6functional_1_dense_2_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
functional_1/dense_2_1/BiasAddBiasAdd&functional_1/dense_2_1/MatMul:output:05functional_1/dense_2_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������ �
functional_1/add_1_2/AddAddV2,functional_1/layer_normalization_1/add_2:z:0'functional_1/dense_2_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :������������������ �
Cfunctional_1/layer_normalization_1_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
1functional_1/layer_normalization_1_2/moments/meanMeanfunctional_1/add_1_2/Add:z:0Lfunctional_1/layer_normalization_1_2/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
9functional_1/layer_normalization_1_2/moments/StopGradientStopGradient:functional_1/layer_normalization_1_2/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
>functional_1/layer_normalization_1_2/moments/SquaredDifferenceSquaredDifferencefunctional_1/add_1_2/Add:z:0Bfunctional_1/layer_normalization_1_2/moments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ �
Gfunctional_1/layer_normalization_1_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
5functional_1/layer_normalization_1_2/moments/varianceMeanBfunctional_1/layer_normalization_1_2/moments/SquaredDifference:z:0Pfunctional_1/layer_normalization_1_2/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
;functional_1/layer_normalization_1_2/Reshape/ReadVariableOpReadVariableOpDfunctional_1_layer_normalization_1_2_reshape_readvariableop_resource*
_output_shapes
: *
dtype0�
2functional_1/layer_normalization_1_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"          �
,functional_1/layer_normalization_1_2/ReshapeReshapeCfunctional_1/layer_normalization_1_2/Reshape/ReadVariableOp:value:0;functional_1/layer_normalization_1_2/Reshape/shape:output:0*
T0*"
_output_shapes
: �
=functional_1/layer_normalization_1_2/Reshape_1/ReadVariableOpReadVariableOpFfunctional_1_layer_normalization_1_2_reshape_1_readvariableop_resource*
_output_shapes
: *
dtype0�
4functional_1/layer_normalization_1_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"          �
.functional_1/layer_normalization_1_2/Reshape_1ReshapeEfunctional_1/layer_normalization_1_2/Reshape_1/ReadVariableOp:value:0=functional_1/layer_normalization_1_2/Reshape_1/shape:output:0*
T0*"
_output_shapes
: o
*functional_1/layer_normalization_1_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
(functional_1/layer_normalization_1_2/addAddV2>functional_1/layer_normalization_1_2/moments/variance:output:03functional_1/layer_normalization_1_2/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
*functional_1/layer_normalization_1_2/RsqrtRsqrt,functional_1/layer_normalization_1_2/add:z:0*
T0*4
_output_shapes"
 :�������������������
(functional_1/layer_normalization_1_2/mulMul.functional_1/layer_normalization_1_2/Rsqrt:y:05functional_1/layer_normalization_1_2/Reshape:output:0*
T0*4
_output_shapes"
 :������������������ �
(functional_1/layer_normalization_1_2/NegNeg:functional_1/layer_normalization_1_2/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
*functional_1/layer_normalization_1_2/mul_1Mul,functional_1/layer_normalization_1_2/Neg:y:0,functional_1/layer_normalization_1_2/mul:z:0*
T0*4
_output_shapes"
 :������������������ �
*functional_1/layer_normalization_1_2/add_1AddV2.functional_1/layer_normalization_1_2/mul_1:z:07functional_1/layer_normalization_1_2/Reshape_1:output:0*
T0*4
_output_shapes"
 :������������������ �
*functional_1/layer_normalization_1_2/mul_2Mulfunctional_1/add_1_2/Add:z:0,functional_1/layer_normalization_1_2/mul:z:0*
T0*4
_output_shapes"
 :������������������ �
*functional_1/layer_normalization_1_2/add_2AddV2.functional_1/layer_normalization_1_2/mul_2:z:0.functional_1/layer_normalization_1_2/add_1:z:0*
T0*4
_output_shapes"
 :������������������ �
>functional_1/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
,functional_1/global_average_pooling1d_1/MeanMean.functional_1/layer_normalization_1_2/add_2:z:0Gfunctional_1/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:��������� �
*functional_1/dense_3_1/Cast/ReadVariableOpReadVariableOp3functional_1_dense_3_1_cast_readvariableop_resource*
_output_shapes

: @*
dtype0�
functional_1/dense_3_1/MatMulMatMul5functional_1/global_average_pooling1d_1/Mean:output:02functional_1/dense_3_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
-functional_1/dense_3_1/BiasAdd/ReadVariableOpReadVariableOp6functional_1_dense_3_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
functional_1/dense_3_1/BiasAddBiasAdd'functional_1/dense_3_1/MatMul:product:05functional_1/dense_3_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
functional_1/dense_3_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
functional_1/dense_3_1/mulMul$functional_1/dense_3_1/beta:output:0'functional_1/dense_3_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@{
functional_1/dense_3_1/SigmoidSigmoidfunctional_1/dense_3_1/mul:z:0*
T0*'
_output_shapes
:���������@�
functional_1/dense_3_1/mul_1Mul'functional_1/dense_3_1/BiasAdd:output:0"functional_1/dense_3_1/Sigmoid:y:0*
T0*'
_output_shapes
:���������@
functional_1/dense_3_1/IdentityIdentity functional_1/dense_3_1/mul_1:z:0*
T0*'
_output_shapes
:���������@�
 functional_1/dense_3_1/IdentityN	IdentityN functional_1/dense_3_1/mul_1:z:0'functional_1/dense_3_1/BiasAdd:output:0$functional_1/dense_3_1/beta:output:0*
T
2**
_gradient_op_typeCustomGradient-2048*<
_output_shapes*
(:���������@:���������@: �
*functional_1/dense_4_1/Cast/ReadVariableOpReadVariableOp3functional_1_dense_4_1_cast_readvariableop_resource*
_output_shapes

:@@*
dtype0�
functional_1/dense_4_1/MatMulMatMul)functional_1/dense_3_1/IdentityN:output:02functional_1/dense_4_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
-functional_1/dense_4_1/BiasAdd/ReadVariableOpReadVariableOp6functional_1_dense_4_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
functional_1/dense_4_1/BiasAddBiasAdd'functional_1/dense_4_1/MatMul:product:05functional_1/dense_4_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
functional_1/dense_4_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
functional_1/dense_4_1/mulMul$functional_1/dense_4_1/beta:output:0'functional_1/dense_4_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@{
functional_1/dense_4_1/SigmoidSigmoidfunctional_1/dense_4_1/mul:z:0*
T0*'
_output_shapes
:���������@�
functional_1/dense_4_1/mul_1Mul'functional_1/dense_4_1/BiasAdd:output:0"functional_1/dense_4_1/Sigmoid:y:0*
T0*'
_output_shapes
:���������@
functional_1/dense_4_1/IdentityIdentity functional_1/dense_4_1/mul_1:z:0*
T0*'
_output_shapes
:���������@�
 functional_1/dense_4_1/IdentityN	IdentityN functional_1/dense_4_1/mul_1:z:0'functional_1/dense_4_1/BiasAdd:output:0$functional_1/dense_4_1/beta:output:0*
T
2**
_gradient_op_typeCustomGradient-2063*<
_output_shapes*
(:���������@:���������@: �
*functional_1/dense_5_1/Cast/ReadVariableOpReadVariableOp3functional_1_dense_5_1_cast_readvariableop_resource*
_output_shapes

:@H*
dtype0�
functional_1/dense_5_1/MatMulMatMul)functional_1/dense_4_1/IdentityN:output:02functional_1/dense_5_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������H�
-functional_1/dense_5_1/BiasAdd/ReadVariableOpReadVariableOp6functional_1_dense_5_1_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0�
functional_1/dense_5_1/BiasAddBiasAdd'functional_1/dense_5_1/MatMul:product:05functional_1/dense_5_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������H�
functional_1/reshape_1/ShapeShape'functional_1/dense_5_1/BiasAdd:output:0*
T0*
_output_shapes
::��t
*functional_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,functional_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,functional_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$functional_1/reshape_1/strided_sliceStridedSlice%functional_1/reshape_1/Shape:output:03functional_1/reshape_1/strided_slice/stack:output:05functional_1/reshape_1/strided_slice/stack_1:output:05functional_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&functional_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :h
&functional_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
$functional_1/reshape_1/Reshape/shapePack-functional_1/reshape_1/strided_slice:output:0/functional_1/reshape_1/Reshape/shape/1:output:0/functional_1/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
functional_1/reshape_1/ReshapeReshape'functional_1/dense_5_1/BiasAdd:output:0-functional_1/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:���������z
IdentityIdentity'functional_1/reshape_1/Reshape:output:0^NoOp*
T0*+
_output_shapes
:����������

NoOpNoOp,^functional_1/dense_1/BiasAdd/ReadVariableOp)^functional_1/dense_1/Cast/ReadVariableOp.^functional_1/dense_1_2/BiasAdd/ReadVariableOp+^functional_1/dense_1_2/Cast/ReadVariableOp.^functional_1/dense_2_1/BiasAdd/ReadVariableOp+^functional_1/dense_2_1/Cast/ReadVariableOp.^functional_1/dense_3_1/BiasAdd/ReadVariableOp+^functional_1/dense_3_1/Cast/ReadVariableOp.^functional_1/dense_4_1/BiasAdd/ReadVariableOp+^functional_1/dense_4_1/Cast/ReadVariableOp.^functional_1/dense_5_1/BiasAdd/ReadVariableOp+^functional_1/dense_5_1/Cast/ReadVariableOp:^functional_1/layer_normalization_1/Reshape/ReadVariableOp<^functional_1/layer_normalization_1/Reshape_1/ReadVariableOp<^functional_1/layer_normalization_1_2/Reshape/ReadVariableOp>^functional_1/layer_normalization_1_2/Reshape_1/ReadVariableOpK^functional_1/multi_head_attention_1/attention_output_1/Cast/ReadVariableOpJ^functional_1/multi_head_attention_1/attention_output_1/add/ReadVariableOp>^functional_1/multi_head_attention_1/key_1/Cast/ReadVariableOp=^functional_1/multi_head_attention_1/key_1/add/ReadVariableOp@^functional_1/multi_head_attention_1/query_1/Cast/ReadVariableOp?^functional_1/multi_head_attention_1/query_1/add/ReadVariableOp@^functional_1/multi_head_attention_1/value_1/Cast/ReadVariableOp?^functional_1/multi_head_attention_1/value_1/add/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:������������������: : :� : : : : : : : : : : : : : : : : : : : : : : 2Z
+functional_1/dense_1/BiasAdd/ReadVariableOp+functional_1/dense_1/BiasAdd/ReadVariableOp2T
(functional_1/dense_1/Cast/ReadVariableOp(functional_1/dense_1/Cast/ReadVariableOp2^
-functional_1/dense_1_2/BiasAdd/ReadVariableOp-functional_1/dense_1_2/BiasAdd/ReadVariableOp2X
*functional_1/dense_1_2/Cast/ReadVariableOp*functional_1/dense_1_2/Cast/ReadVariableOp2^
-functional_1/dense_2_1/BiasAdd/ReadVariableOp-functional_1/dense_2_1/BiasAdd/ReadVariableOp2X
*functional_1/dense_2_1/Cast/ReadVariableOp*functional_1/dense_2_1/Cast/ReadVariableOp2^
-functional_1/dense_3_1/BiasAdd/ReadVariableOp-functional_1/dense_3_1/BiasAdd/ReadVariableOp2X
*functional_1/dense_3_1/Cast/ReadVariableOp*functional_1/dense_3_1/Cast/ReadVariableOp2^
-functional_1/dense_4_1/BiasAdd/ReadVariableOp-functional_1/dense_4_1/BiasAdd/ReadVariableOp2X
*functional_1/dense_4_1/Cast/ReadVariableOp*functional_1/dense_4_1/Cast/ReadVariableOp2^
-functional_1/dense_5_1/BiasAdd/ReadVariableOp-functional_1/dense_5_1/BiasAdd/ReadVariableOp2X
*functional_1/dense_5_1/Cast/ReadVariableOp*functional_1/dense_5_1/Cast/ReadVariableOp2v
9functional_1/layer_normalization_1/Reshape/ReadVariableOp9functional_1/layer_normalization_1/Reshape/ReadVariableOp2z
;functional_1/layer_normalization_1/Reshape_1/ReadVariableOp;functional_1/layer_normalization_1/Reshape_1/ReadVariableOp2z
;functional_1/layer_normalization_1_2/Reshape/ReadVariableOp;functional_1/layer_normalization_1_2/Reshape/ReadVariableOp2~
=functional_1/layer_normalization_1_2/Reshape_1/ReadVariableOp=functional_1/layer_normalization_1_2/Reshape_1/ReadVariableOp2�
Jfunctional_1/multi_head_attention_1/attention_output_1/Cast/ReadVariableOpJfunctional_1/multi_head_attention_1/attention_output_1/Cast/ReadVariableOp2�
Ifunctional_1/multi_head_attention_1/attention_output_1/add/ReadVariableOpIfunctional_1/multi_head_attention_1/attention_output_1/add/ReadVariableOp2~
=functional_1/multi_head_attention_1/key_1/Cast/ReadVariableOp=functional_1/multi_head_attention_1/key_1/Cast/ReadVariableOp2|
<functional_1/multi_head_attention_1/key_1/add/ReadVariableOp<functional_1/multi_head_attention_1/key_1/add/ReadVariableOp2�
?functional_1/multi_head_attention_1/query_1/Cast/ReadVariableOp?functional_1/multi_head_attention_1/query_1/Cast/ReadVariableOp2�
>functional_1/multi_head_attention_1/query_1/add/ReadVariableOp>functional_1/multi_head_attention_1/query_1/add/ReadVariableOp2�
?functional_1/multi_head_attention_1/value_1/Cast/ReadVariableOp?functional_1/multi_head_attention_1/value_1/Cast/ReadVariableOp2�
>functional_1/multi_head_attention_1/value_1/add/ReadVariableOp>functional_1/multi_head_attention_1/value_1/add/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:IE
#
_output_shapes
:� 

_user_specified_name1929:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:b ^
4
_output_shapes"
 :������������������
&
_user_specified_namekeras_tensor8
!__inference_internal_grad_fn_2447CustomGradient-20488
!__inference_internal_grad_fn_2474CustomGradient-2063"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serve�
H
keras_tensor8
serve_keras_tensor:0������������������@
output_04
StatefulPartitionedCall:0���������tensorflow/serving/predict*�
serving_default�
R
keras_tensorB
serving_default_keras_tensor:0������������������B
output_06
StatefulPartitionedCall_1:0���������tensorflow/serving/predict:�+
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures"
_generic_user_object
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27"
trackable_list_wrapper
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
 21
"22
#23"
trackable_list_wrapper
<
0
1
2
!3"
trackable_list_wrapper
�
$0
%1
&2
'3
(4
)5
*6
+7
,8
-9
.10
/11
012
113
214
315
416
517
618
719
820
921
:22
;23"
trackable_list_wrapper
 "
trackable_list_wrapper
�
<trace_02�
__inference___call___2087�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0
keras_tensor������������������z<trace_0
7
	=serve
>serving_default"
signature_map
: 2dense/kernel
: 2
dense/bias
7:5 
 2!multi_head_attention/query/kernel
1:/
 2multi_head_attention/query/bias
5:3 
 2multi_head_attention/key/kernel
/:-
 2multi_head_attention/key/bias
7:5 
 2!multi_head_attention/value/kernel
1:/
 2multi_head_attention/value/bias
B:@
  2,multi_head_attention/attention_output/kernel
8:6 2*multi_head_attention/attention_output/bias
/:-	2#seed_generator/seed_generator_state
':% 2layer_normalization/gamma
&:$ 2layer_normalization/beta
!:	 �2dense_1/kernel
:�2dense_1/bias
!:	� 2dense_2/kernel
: 2dense_2/bias
1:/	2%seed_generator_1/seed_generator_state
):' 2layer_normalization_1/gamma
(:& 2layer_normalization_1/beta
 : @2dense_3/kernel
:@2dense_3/bias
1:/	2%seed_generator_2/seed_generator_state
 :@@2dense_4/kernel
:@2dense_4/bias
1:/	2%seed_generator_3/seed_generator_state
 :@H2dense_5/kernel
:H2dense_5/bias
5:3 
 2multi_head_attention/key/kernel
B:@
  2,multi_head_attention/attention_output/kernel
:�2dense_1/bias
):' 2layer_normalization_1/gamma
(:& 2layer_normalization_1/beta
 :@@2dense_4/kernel
':% 2layer_normalization/gamma
 : @2dense_3/kernel
7:5 
 2!multi_head_attention/query/kernel
8:6 2*multi_head_attention/attention_output/bias
:@2dense_3/bias
1:/
 2multi_head_attention/query/bias
&:$ 2layer_normalization/beta
/:-
 2multi_head_attention/key/bias
7:5 
 2!multi_head_attention/value/kernel
:@2dense_4/bias
1:/
 2multi_head_attention/value/bias
!:	 �2dense_1/kernel
:H2dense_5/bias
!:	� 2dense_2/kernel
 :@H2dense_5/kernel
: 2dense/kernel
: 2
dense/bias
: 2dense_2/bias
�
?	capture_2B�
__inference___call___2087keras_tensor"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z?	capture_2
�
?	capture_2B�
+__inference_signature_wrapper___call___2143keras_tensor"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 !

kwonlyargs�
jkeras_tensor
kwonlydefaults
 
annotations� *
 z?	capture_2
�
?	capture_2B�
+__inference_signature_wrapper___call___2198keras_tensor"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 !

kwonlyargs�
jkeras_tensor
kwonlydefaults
 
annotations� *
 z?	capture_2
J
Constjtf.TrackableConstant
<b:
functional_1/dense_3_1/beta:0__inference___call___2087
?b=
 functional_1/dense_3_1/BiasAdd:0__inference___call___2087
<b:
functional_1/dense_4_1/beta:0__inference___call___2087
?b=
 functional_1/dense_4_1/BiasAdd:0__inference___call___2087�
__inference___call___2087�	?
 "#B�?
8�5
3�0
keras_tensor������������������
� "%�"
unknown����������
!__inference_internal_grad_fn_2447�@A~�{
t�q

 
(�%
result_grads_0���������@
(�%
result_grads_1���������@
�
result_grads_2 
� ">�;

 
"�
tensor_1���������@
�
tensor_2 �
!__inference_internal_grad_fn_2474�BC~�{
t�q

 
(�%
result_grads_0���������@
(�%
result_grads_1���������@
�
result_grads_2 
� ">�;

 
"�
tensor_1���������@
�
tensor_2 �
+__inference_signature_wrapper___call___2143�	?
 "#R�O
� 
H�E
C
keras_tensor3�0
keras_tensor������������������"7�4
2
output_0&�#
output_0����������
+__inference_signature_wrapper___call___2198�	?
 "#R�O
� 
H�E
C
keras_tensor3�0
keras_tensor������������������"7�4
2
output_0&�#
output_0���������