/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include "node_color_temperature.h"
#include "exception.h"

ColorTemperatureNode::ColorTemperatureNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _adj_value_param(ADJUSTMENT_RANGE[0], ADJUSTMENT_RANGE[1])
{
}

void ColorTemperatureNode::create_node()
{
    if(_node)
        return;

    _adj_value_param.create_array(_graph , VX_TYPE_INT32, _batch_size);

    _node = vxExtrppNode_ColorTemperaturebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _adj_value_param.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the color temp batch (vxExtrppNode_ColorTemperaturebatchPD) node failed: "+ TOSTR(status))

}

void ColorTemperatureNode::init(int adjustment)
{
    _adj_value_param.set_param(adjustment);
}

void ColorTemperatureNode::init(IntParam* adjustment)
{
    _adj_value_param.set_param(core(adjustment));
}

void ColorTemperatureNode::update_node()
{
    _adj_value_param.update_array();
}
