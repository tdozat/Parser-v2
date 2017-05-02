#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Timothy Dozat
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

colors = {
  None: '\033[0m',
  'bold': '\033[1m',
  'italic': '\033[3m',
  'uline': '\033[4m',
  'blink': '\033[5m',
  'hlight': '\033[7m',
  
  'black': '\033[30m',
  'red': '\033[31m',
  'green': '\033[32m',
  'yellow': '\033[33m',
  'blue': '\033[34m',
  'magenta': '\033[35m',
  'cyan': '\033[36m',
  'white': '\033[37m',
  
  'black_hlight': '\033[40m',
  'red_hlight': '\033[41m',
  'green_hlight': '\033[42m',
  'yellow_hlight': '\033[43m',
  'blue_hlight': '\033[44m',
  'magenta_hlight': '\033[45m',
  'cyan_hlight': '\033[46m',
  'white_hlight': '\033[47m',
  
  'bright_black': '\033[90m',
  'bright_red': '\033[91m',
  'bright_green': '\033[92m',
  'bright_yellow': '\033[93m',
  'bright_blue': '\033[94m',
  'bright_magenta': '\033[95m',
  'bright_cyan': '\033[96m',
  'bright_white': '\033[97m',
  
  'bright_black_hlight': '\033[100m',
  'bright_red_hlight': '\033[101m',
  'bright_green_hlight': '\033[102m',
  'bright_orange_hlight': '\033[103m',
  'bright_blue_hlight': '\033[1010m',
  'bright_magenta_hlight': '\033[105m',
  'bright_cyan_hlight': '\033[106m',
  'bright_white_hlight': '\033[107m',
}

def ctext(text, *color_list):
  return ''.join(colors[color] for color in color_list) + text + colors[None]
def color_pattern(text1, text2, *color_list):
  multicolor = ''.join(colors[color] for color in color_list)
  return multicolor + colors['bold'] + text1 + colors[None] + ' ' + multicolor + colors['uline'] + text2 + colors[None]
