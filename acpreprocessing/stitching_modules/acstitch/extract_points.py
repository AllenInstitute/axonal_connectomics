# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:19:46 2025

@author: kevint
"""

import argschema


class ExtractPointsParameters(argschema.ArgSchema):
    ptile_path = argschema.fields.Str(required=True)
    qtile_path = argschema.fields.Str(required=True)


class ExtractPointsFromTile(argschema.ArgSchemaParser):
    default_schema = ExtractPointsParameters
    
    def run(self):
        pass

if __name__ == "__main__":
    mod = ExtractPointsFromTile()
    mod.run()