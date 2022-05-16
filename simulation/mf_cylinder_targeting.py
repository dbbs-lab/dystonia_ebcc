"""
Script required for mossy fiber cylinder targeting (see in .json config file "devices" -> "CS")
"""

import numpy as np
from bsb.core import register_cell_targetting

def mf_cylinder(self):
    target_cells = np.empty((0, 2))
    id_map = np.empty(0)

    ps = self.scaffold.get_placement_set("glomerulus")
    pos = ps.positions[:, [0, 2]]
    target_cells = np.vstack((target_cells, pos))
    id_map = np.concatenate((id_map, ps.identifiers))

    if not hasattr(self, "origin"):
        x = self.scaffold.configuration.X
        z = self.scaffold.configuration.Z
        origin = np.array((x, z))
    else:
        origin = np.array(self.origin)
    in_range_mask = (
        np.sum((target_cells[:, [0, 1]] - origin) ** 2, axis=1) < self.radius ** 2
    )

    id_map_glom = id_map[in_range_mask].astype(int).tolist()
    conn_glom_mf = self.scaffold.get_connectivity_set("mossy_to_glomerulus")
    ids_glom_all = conn_glom_mf.to_identifiers
    ids_mf_all = conn_glom_mf.from_identifiers
    mf_ids = []
    for id in range(len(ids_glom_all)):
        if ids_glom_all[id] in id_map_glom:
            mf_ids.append(ids_mf_all[id])
    mf_ids = np.unique(mf_ids)
    return mf_ids

register_cell_targetting("mf_cylinder", mf_cylinder)
