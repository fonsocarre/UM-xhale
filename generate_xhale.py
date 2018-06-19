import h5py as h5
import numpy as np
import os
import sharpy.utils.algebra as algebra

case_name = 'xhale_rrv-6b'
route = os.path.dirname(os.path.realpath(__file__)) + '/'

flow = ['BeamLoader',
        'AerogridLoader',
        # 'NonLinearStatic',
        # 'StaticCoupled',
        # 'BeamLoads',
        'BeamPlot',
        'AerogridPlot'
        ]

u_inf = 10
rho = 1.225
alpha = 0*np.pi/180
beta = 0
gravity = 'on'
gravity_value = 9.81
cs_deflection = 0.0*np.pi/180
thrust = 0
sigma = 1

gust_intensity = 0.0
n_step = 1
relaxation_factor = 0.4
tolerance = 1e-8

span_section = 1.0
dihedral_outer = 10*np.pi/180

ea_main = 0.288
cg_main = 0.25

length_centre_tail = 0.9
length_outer_tail = 0.65
span_tail = 0.24
ea_tail = 0.25

main_wing_chord = 0.2
tail_chord = 0.11

n_sections = 3

# DISCRETISATION
# spatial discretisation
m = 5
n_elem_multiplier = 1
n_elem_section = 3
n_elem_centre_tail = 2
n_elem_outer_tail = 2
n_elem_tail = 1
n_elem_main = int(n_sections*n_elem_section*n_elem_multiplier)
n_surfaces = 16

# temporal discretisation
physical_time = 30
tstep_factor = 1.0
dt = 1.0/m/u_inf*tstep_factor
n_tstep = round(physical_time/dt)
n_tstep = 1500

# beam processing
n_node_elem = 3
span_main = n_sections*span_section

# total number of elements
n_elem = 0
n_elem += n_elem_main
n_elem += n_elem_main
n_elem += n_elem_centre_tail
n_elem += n_elem_tail
n_elem += n_elem_tail
n_elem += n_elem_outer_tail
n_elem += n_elem_tail
n_elem += n_elem_tail
n_elem += n_elem_outer_tail
n_elem += n_elem_tail
n_elem += n_elem_tail
n_elem += n_elem_outer_tail
n_elem += n_elem_tail
n_elem += n_elem_tail
n_elem += n_elem_outer_tail
n_elem += n_elem_tail
n_elem += n_elem_tail

# number of nodes per part
n_node_section = n_elem_section*(n_node_elem - 1) + 1
n_node_main = n_elem_main*(n_node_elem - 1) + 1
n_node_centre_tail = n_elem_centre_tail*(n_node_elem - 1) + 1
n_node_tail = n_elem_tail*(n_node_elem - 1) + 1
n_node_outer_tail = n_elem_outer_tail*(n_node_elem - 1) + 1

# total number of nodes
n_node = 0
n_node += n_node_main + n_node_main - 1
n_node += n_node_centre_tail - 1
n_node += n_node_tail - 1
n_node += n_node_tail - 1
n_node += n_node_outer_tail - 1
n_node += n_node_tail - 1
n_node += n_node_tail - 1
n_node += n_node_outer_tail - 1
n_node += n_node_tail - 1
n_node += n_node_tail - 1
n_node += n_node_outer_tail - 1
n_node += n_node_tail - 1
n_node += n_node_tail - 1
n_node += n_node_outer_tail - 1
n_node += n_node_tail - 1
n_node += n_node_tail - 1

# stiffness and mass matrices
n_stiffness = 5
n_mass = 5

# PLACEHOLDERS
# beam
x = np.zeros((n_node, ))
y = np.zeros((n_node, ))
z = np.zeros((n_node, ))
stiffness_db = np.zeros((n_stiffness, 6, 6))
mass_db = np.zeros((n_mass, 6, 6))
structural_twist = np.zeros_like(x)
beam_number = np.zeros((n_elem, ), dtype=int)
frame_of_reference_delta = np.zeros((n_elem, n_node_elem, 3))
conn = np.zeros((n_elem, n_node_elem), dtype=int)
elem_stiffness = np.zeros((n_elem, ), dtype=int)
elem_mass = np.zeros((n_elem, ), dtype=int)
boundary_conditions = np.zeros((n_node, ), dtype=int)
app_forces = np.zeros((n_node, 6))
n_lumped_mass = 0
lumped_mass_nodes = np.zeros((n_lumped_mass, ), dtype=int)
lumped_mass = np.zeros((n_lumped_mass, ))
lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
lumped_mass_position = np.zeros((n_lumped_mass, 3))

end_nodesL = np.zeros((n_sections,), dtype=int)
end_nodesR = np.zeros((n_sections,), dtype=int)
end_elementsL = np.zeros((n_sections,), dtype=int)
end_elementsR = np.zeros((n_sections,), dtype=int)
end_tails_nodesL = np.zeros((2, ), dtype=int)
end_tails_elementsL = np.zeros((2, ), dtype=int)
end_tails_nodesR = np.zeros((2, ), dtype=int)
end_tails_elementsR = np.zeros((2, ), dtype=int)
end_of_centre_tail_node  = 0
end_of_centre_tail_elem  = 0

end_tip_tail_nodeC = np.zeros((2, ), dtype=int)
end_tip_tail_elemC = np.zeros((2, ), dtype=int)

tail_beam_numbersR = np.zeros((2, 3)) # 0=centre spar, 1=R tail, 2=L tail
tail_beam_numbersL = np.zeros((2, 3)) # 0=centre spar, 1=R tail, 2=L tail
tail_beam_numbersC = np.zeros((3, ))

# aero
airfoil_distribution = np.zeros((n_elem, n_node_elem), dtype=int)
surface_distribution = np.zeros((n_elem,), dtype=int) - 1
surface_m = np.zeros((n_surfaces, ), dtype=int)
m_distribution = 'uniform'
aero_node = np.zeros((n_node,), dtype=bool)
twist = np.zeros((n_elem, n_node_elem))
chord = np.zeros((n_elem, n_node_elem,))
elastic_axis = np.zeros((n_elem, n_node_elem,))


# FUNCTIONS-------------------------------------------------------------
def clean_test_files():
    fem_file_name = route + '/' + case_name + '.fem.h5'
    if os.path.isfile(fem_file_name):
        os.remove(fem_file_name)

    dyn_file_name = route + '/' + case_name + '.dyn.h5'
    if os.path.isfile(dyn_file_name):
        os.remove(dyn_file_name)

    aero_file_name = route + '/' + case_name + '.aero.h5'
    if os.path.isfile(aero_file_name):
        os.remove(aero_file_name)

    solver_file_name = route + '/' + case_name + '.solver.txt'
    if os.path.isfile(solver_file_name):
        os.remove(solver_file_name)

    flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
    if os.path.isfile(flightcon_file_name):
        os.remove(flightcon_file_name)

def read_beam_data(filename='inputs/beam_properties.xlsx'):
    import pandas as pd
    # mass
    mass_sheet = pd.read_excel(filename, sheetname='mass', header=1, skip_rows=1, index_col=0)
    # remove units
    mass_sheet = mass_sheet.drop(['[-]'])
    mass_data = dict()
    for index, row in mass_sheet.iterrows():
        # import pdb; pdb.set_trace()
        mass_data[index] = np.zeros((6, 6))
        mass_data[index][np.diag_indices(3)] = mass_sheet['mass'][index]
        mass_data[index][3, 3] = mass_sheet['ixx'][index]
        mass_data[index][4, 4] = mass_sheet['iyy'][index]
        mass_data[index][5, 5] = mass_sheet['izz'][index]

    # stiffness
    stiff_sheet = pd.read_excel(filename, sheetname='stiffness', header=1, skip_rows=1, index_col=0)
    # remove units
    stiff_sheet = stiff_sheet.drop(['[-]'])
    stiff_data = dict()
    for index, row in stiff_sheet.iterrows():
        # import pdb; pdb.set_trace()
        stiff_data[index] = np.zeros((6, 6))
        stiff_data[index][0, 0] = stiff_sheet['ea'][index]
        stiff_data[index][1, 1] = stiff_sheet['gay'][index]
        stiff_data[index][2, 2] = stiff_sheet['gaz'][index]
        stiff_data[index][3, 3] = stiff_sheet['gj'][index]
        stiff_data[index][4, 4] = stiff_sheet['eiy'][index]
        stiff_data[index][5, 5] = stiff_sheet['eiz'][index]

    return mass_data, stiff_data

def read_lumped_mass_data(filename='inputs/lumped_mass.xlsx'):



def generate_fem():
    global end_of_centre_tail_node, end_of_centre_tail_elem

    mass_data, stiff_data = read_beam_data()

    mass_db[0, ...] = mass_data['inboard']
    mass_db[1, ...] = mass_data['outboard']
    mass_db[2, ...] = mass_data['dihedral']
    mass_db[3, ...] = mass_data['boom']
    mass_db[4, ...] = mass_data['tail']

    stiffness_db[0, ...] = stiff_data['inboard']
    stiffness_db[1, ...] = stiff_data['outboard']
    stiffness_db[2, ...] = stiff_data['dihedral']
    stiffness_db[3, ...] = stiff_data['boom']
    stiffness_db[4, ...] = stiff_data['tail']

    we = 0
    wn = 0

    # SECTION 0R
    beam_number[we:we + n_elem_section] = 0
    y[wn:wn + n_node_section] = np.linspace(0.0, span_section, n_node_section)
    for ielem in range(n_elem_section):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    elem_stiffness[we:we + n_elem_section] = 0
    elem_mass[we:we + n_elem_section] = 0
    boundary_conditions[0] = 1
    we += n_elem_section
    wn += n_node_section
    end_nodesR[0] = wn - 1
    end_elementsR[0] = we - 1

    # SECTION 1R
    beam_number[we:we + n_elem_section] = 1
    y[wn:wn + n_node_section - 1] = y[wn - 1] + np.linspace(0.0, span_section, n_node_section)[1:]
    for ielem in range(n_elem_section):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    elem_stiffness[we:we + n_elem_section] = 1
    elem_mass[we:we + n_elem_section] = 1
    we += n_elem_section
    wn += n_node_section - 1
    end_nodesR[1] = wn - 1
    end_elementsR[1] = we - 1

    # SECTION 2R
    beam_number[we:we + n_elem_section] = 2
    y[wn:wn + n_node_section - 1] = y[wn - 1] + np.linspace(0.0, span_section, n_node_section)[1:]
    for ielem in range(n_elem_section):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    elem_stiffness[we:we + n_elem_section] = 2
    elem_mass[we:we + n_elem_section] = 2
    boundary_conditions[wn + n_node_section - 1 - 1] = -1
    we += n_elem_section
    wn += n_node_section - 1
    end_nodesR[2] = wn - 1
    end_elementsR[2] = we - 1

    # # SECTION 3R
    # beam_number[we:we + n_elem_section] = 3
    # y[wn:wn + n_node_section - 1] = y[wn - 1] + np.linspace(0.0,
    #                                                         np.cos(dihedral_outer)*span_section,
    #                                                         n_node_section)[1:]
    # z[wn:wn + n_node_section - 1] = z[wn - 1] + np.linspace(0.0,
    #                                                         np.sin(dihedral_outer)*span_section,
    #                                                         n_node_section)[1:]
    # for ielem in range(n_elem_section):
    #     conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
    #     for inode in range(n_node_elem):
    #         frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    # elem_stiffness[we:we + n_elem_section] = 0
    # elem_mass[we:we + n_elem_section] = 0
    # boundary_conditions[wn + n_node_section - 1 - 1] = -1
    # we += n_elem_section
    # wn += n_node_section - 1
    # end_nodesR[3] = wn - 1
    # end_elementsR[3] = we - 1

    # SECTION 0L
    beam_number[we:we + n_elem_section] = 3
    y[wn:wn + n_node_section - 1] = np.linspace(0.0, -span_section, n_node_section)[1:]
    for ielem in range(n_elem_section):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    conn[we, 0] = 0
    elem_stiffness[we:we + n_elem_section] = 0
    elem_mass[we:we + n_elem_section] = 0
    we += n_elem_section
    wn += n_node_section - 1
    end_nodesL[0] = wn - 1
    end_elementsL[0] = we - 1

    # SECTION 1L
    beam_number[we:we + n_elem_section] = 4
    y[wn:wn + n_node_section - 1] = y[wn - 1] + np.linspace(0.0, -span_section, n_node_section)[1:]
    for ielem in range(n_elem_section):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    elem_stiffness[we:we + n_elem_section] = 0
    elem_mass[we:we + n_elem_section] = 0
    we += n_elem_section
    wn += n_node_section - 1
    end_nodesL[1] = wn - 1
    end_elementsL[1] = we - 1

    # SECTION 2L
    beam_number[we:we + n_elem_section] = 5
    y[wn:wn + n_node_section - 1] = y[wn - 1] + np.linspace(0.0, -span_section, n_node_section)[1:]
    for ielem in range(n_elem_section):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    elem_stiffness[we:we + n_elem_section] = 0
    elem_mass[we:we + n_elem_section] = 0
    boundary_conditions[wn + n_node_section - 1 - 1] = -1
    we += n_elem_section
    wn += n_node_section - 1
    end_nodesL[2] = wn - 1
    end_elementsL[2] = we - 1

    # # SECTION 3L
    # beam_number[we:we + n_elem_section] = 7
    # y[wn:wn + n_node_section - 1] = y[wn - 1] + np.linspace(0.0, -np.cos(dihedral_outer)*span_section, n_node_section)[1:]
    # z[wn:wn + n_node_section - 1] = z[wn - 1] + np.linspace(0.0, np.sin(dihedral_outer)*span_section, n_node_section)[1:]
    # for ielem in range(n_elem_section):
    #     conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
    #     for inode in range(n_node_elem):
    #         frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    # elem_stiffness[we:we + n_elem_section] = 0
    # elem_mass[we:we + n_elem_section] = 0
    # boundary_conditions[wn + n_node_section - 1 - 1] = -1
    # we += n_elem_section
    # wn += n_node_section - 1
    # end_nodesL[3] = wn - 1
    # end_elementsL[3] = we - 1

    # centre tail
    beam_number[we:we + n_elem_centre_tail] = 6
    tail_beam_numbersC[0] = 6
    x[wn:wn + n_node_centre_tail - 1] = np.linspace(0.0, length_centre_tail, n_node_centre_tail)[1:]
    for ielem in range(n_elem_centre_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
    conn[we, 0] = 0
    elem_stiffness[we:we + n_elem_centre_tail] = 3
    elem_mass[we:we + n_elem_centre_tail] = 3
    we += n_elem_centre_tail
    wn += n_node_centre_tail - 1
    end_of_centre_tail_node = wn - 1
    end_of_centre_tail_elem = we

    beam_number[we:we + n_elem_tail] = 7
    tail_beam_numbersC[1] = 7
    x[wn:wn + n_node_tail - 1] = x[wn - 1]
    y[wn:wn + n_node_tail - 1] = y[wn - 1] + np.linspace(0.0, span_tail, n_node_tail)[1:]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    elem_stiffness[we:we + n_elem_tail] = 4
    elem_mass[we:we + n_elem_tail] = 4
    boundary_conditions[wn + n_node_tail - 1 - 1] = -1
    end_tip_tail_nodeC[0] = wn + n_node_tail - 1 - 1
    end_tip_tail_elemC[0] = we + n_elem_tail - 1
    we += n_elem_tail
    wn += n_node_tail - 1

    beam_number[we:we + n_elem_tail] = 8
    tail_beam_numbersC[2] = 8
    x[wn:wn + n_node_tail - 1] = x[end_of_centre_tail_node]
    y[wn:wn + n_node_tail - 1] = y[end_of_centre_tail_node] + np.linspace(0.0, -span_tail, n_node_tail)[1:]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    conn[we, 0] = end_of_centre_tail_node
    elem_stiffness[we:we + n_elem_tail] = 4
    elem_mass[we:we + n_elem_tail] = 4
    boundary_conditions[wn + n_node_tail - 1 -1] = -1
    end_tip_tail_nodeC[1] = wn + n_node_tail - 1 - 1
    end_tip_tail_elemC[1] = we + n_elem_tail - 1
    # import pdb; pdb.set_trace()
    we += n_elem_tail
    wn += n_node_tail - 1

    # outer tail 0R
    beam_number[we:we + n_elem_outer_tail] = 9
    tail_beam_numbersR[0,0] = 9
    x[wn:wn + n_node_outer_tail - 1] = np.linspace(0.0, length_outer_tail, n_node_outer_tail)[1:]
    y[wn:wn + n_node_outer_tail - 1] = y[end_nodesR[0]]
    for ielem in range(n_elem_outer_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
    conn[we, 0] = end_nodesR[0]
    elem_stiffness[we:we + n_elem_centre_tail] = 3
    elem_mass[we:we + n_elem_centre_tail] = 3
    we += n_elem_centre_tail
    wn += n_node_centre_tail - 1
    end_tails_nodesR[0] = wn - 1
    end_tails_elementsR[0] = we - 1

    beam_number[we:we + n_elem_tail] = 10
    tail_beam_numbersR[0,1] = 10
    x[wn:wn + n_node_tail - 1] = x[end_tails_nodesR[0]]
    y[wn:wn + n_node_tail - 1] = y[end_tails_nodesR[0]] + np.linspace(0.0, span_tail, n_node_tail)[1:]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    elem_stiffness[we:we + n_elem_tail] = 4
    elem_mass[we:we + n_elem_tail] = 4
    boundary_conditions[wn + n_node_tail - 1 -1] = -1
    we += n_elem_tail
    wn += n_node_tail - 1

    beam_number[we:we + n_elem_tail] = 11
    tail_beam_numbersR[0,2] = 11
    x[wn:wn + n_node_tail - 1] = x[end_tails_nodesR[0]]
    y[wn:wn + n_node_tail - 1] = y[end_tails_nodesR[0]] + np.linspace(0.0, -span_tail, n_node_tail)[1:]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    conn[we, 0] = end_tails_nodesR[0]
    elem_stiffness[we:we + n_elem_tail] = 4
    elem_mass[we:we + n_elem_tail] = 4
    boundary_conditions[wn + n_node_tail - 1 -1] = -1
    we += n_elem_tail
    wn += n_node_tail - 1

    # outer tail 1R
    beam_number[we:we + n_elem_outer_tail] = 12
    tail_beam_numbersR[1,0] = 12
    x[wn:wn + n_node_outer_tail - 1] = np.linspace(0.0, length_outer_tail, n_node_outer_tail)[1:]
    y[wn:wn + n_node_outer_tail - 1] = y[end_nodesR[1]]
    for ielem in range(n_elem_outer_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
    conn[we, 0] = end_nodesR[1]
    elem_stiffness[we:we + n_elem_centre_tail] = 3
    elem_mass[we:we + n_elem_centre_tail] = 3
    we += n_elem_centre_tail
    wn += n_node_centre_tail - 1
    end_tails_nodesR[1] = wn - 1
    end_tails_elementsR[1] = we - 1

    beam_number[we:we + n_elem_tail] = 13
    tail_beam_numbersR[1,1] = 13
    x[wn:wn + n_node_tail - 1] = x[end_tails_nodesR[1]]
    y[wn:wn + n_node_tail - 1] = y[end_tails_nodesR[1]] + np.linspace(0.0, span_tail, n_node_tail)[1:]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    elem_stiffness[we:we + n_elem_tail] = 4
    elem_mass[we:we + n_elem_tail] = 4
    boundary_conditions[wn + n_node_tail - 1 -1] = -1
    we += n_elem_tail
    wn += n_node_tail - 1

    beam_number[we:we + n_elem_tail] = 14
    tail_beam_numbersR[1,2] = 14
    x[wn:wn + n_node_tail - 1] = x[end_tails_nodesR[1]]
    y[wn:wn + n_node_tail - 1] = y[end_tails_nodesR[1]] + np.linspace(0.0, -span_tail, n_node_tail)[1:]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    conn[we, 0] = end_tails_nodesR[1]
    elem_stiffness[we:we + n_elem_tail] = 4
    elem_mass[we:we + n_elem_tail] = 4
    boundary_conditions[wn + n_node_tail - 1 -1] = -1
    we += n_elem_tail
    wn += n_node_tail - 1

    # # outer tail 2R
    # beam_number[we:we + n_elem_outer_tail] = 15
    # tail_beam_numbersR[2,0] = 15
    # x[wn:wn + n_node_outer_tail - 1] = np.linspace(0.0, length_outer_tail, n_node_outer_tail)[1:]
    # y[wn:wn + n_node_outer_tail - 1] = y[end_nodesR[2]]
    # for ielem in range(n_elem_outer_tail):
    #     conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
    #     for inode in range(n_node_elem):
    #         frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
    # conn[we, 0] = end_nodesR[2]
    # elem_stiffness[we:we + n_elem_centre_tail] = 3
    # elem_mass[we:we + n_elem_centre_tail] = 3
    # we += n_elem_centre_tail
    # wn += n_node_centre_tail - 1
    # end_tails_nodesR[2] = wn - 1
    # end_tails_elementsR[2] = we - 1
    #
    # beam_number[we:we + n_elem_tail] = 16
    # tail_beam_numbersR[2,1] = 16
    # x[wn:wn + n_node_tail - 1] = x[end_tails_nodesR[2]]
    # y[wn:wn + n_node_tail - 1] = y[end_tails_nodesR[2]] + np.linspace(0.0, span_tail, n_node_tail)[1:]
    # for ielem in range(n_elem_tail):
    #     conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
    #     for inode in range(n_node_elem):
    #         frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    # elem_stiffness[we:we + n_elem_tail] = 4
    # elem_mass[we:we + n_elem_tail] = 4
    # boundary_conditions[wn + n_node_tail - 1 -1] = -1
    # we += n_elem_tail
    # wn += n_node_tail - 1
    #
    # beam_number[we:we + n_elem_tail] = 17
    # tail_beam_numbersR[2,2] = 17
    # x[wn:wn + n_node_tail - 1] = x[end_tails_nodesR[2]]
    # y[wn:wn + n_node_tail - 1] = y[end_tails_nodesR[2]] + np.linspace(0.0, -span_tail, n_node_tail)[1:]
    # for ielem in range(n_elem_tail):
    #     conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
    #     for inode in range(n_node_elem):
    #         frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    # conn[we, 0] = end_tails_nodesR[2]
    # elem_stiffness[we:we + n_elem_tail] = 4
    # elem_mass[we:we + n_elem_tail] = 4
    # boundary_conditions[wn + n_node_tail - 1 -1] = -1
    # we += n_elem_tail
    # wn += n_node_tail - 1

    # outer tail 0L
    beam_number[we:we + n_elem_outer_tail] = 15
    tail_beam_numbersL[0,0] = 15
    x[wn:wn + n_node_outer_tail - 1] = np.linspace(0.0, length_outer_tail, n_node_outer_tail)[1:]
    y[wn:wn + n_node_outer_tail - 1] = y[end_nodesL[0]]
    for ielem in range(n_elem_outer_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
    conn[we, 0] = end_nodesL[0]
    elem_stiffness[we:we + n_elem_centre_tail] = 3
    elem_mass[we:we + n_elem_centre_tail] = 3
    we += n_elem_centre_tail
    wn += n_node_centre_tail - 1
    end_tails_nodesL[0] = wn - 1
    end_tails_elementsL[0] = we - 1

    beam_number[we:we + n_elem_tail] = 16
    tail_beam_numbersL[0,1] = 16
    x[wn:wn + n_node_tail - 1] = x[end_tails_nodesL[0]]
    y[wn:wn + n_node_tail - 1] = y[end_tails_nodesL[0]] + np.linspace(0.0, span_tail, n_node_tail)[1:]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    elem_stiffness[we:we + n_elem_tail] = 4
    elem_mass[we:we + n_elem_tail] = 4
    boundary_conditions[wn + n_node_tail - 1 -1] = -1
    we += n_elem_tail
    wn += n_node_tail - 1

    beam_number[we:we + n_elem_tail] = 17
    tail_beam_numbersL[0,2] = 17
    x[wn:wn + n_node_tail - 1] = x[end_tails_nodesL[0]]
    y[wn:wn + n_node_tail - 1] = y[end_tails_nodesL[0]] + np.linspace(0.0, -span_tail, n_node_tail)[1:]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    conn[we, 0] = end_tails_nodesL[0]
    elem_stiffness[we:we + n_elem_tail] = 4
    elem_mass[we:we + n_elem_tail] = 4
    boundary_conditions[wn + n_node_tail - 1 -1] = -1
    we += n_elem_tail
    wn += n_node_tail - 1

    # outer tail 1L
    beam_number[we:we + n_elem_outer_tail] = 18
    tail_beam_numbersL[1,0] = 18
    x[wn:wn + n_node_outer_tail - 1] = np.linspace(0.0, length_outer_tail, n_node_outer_tail)[1:]
    y[wn:wn + n_node_outer_tail - 1] = y[end_nodesL[1]]
    for ielem in range(n_elem_outer_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
    conn[we, 0] = end_nodesL[1]
    elem_stiffness[we:we + n_elem_centre_tail] = 3
    elem_mass[we:we + n_elem_centre_tail] = 3
    we += n_elem_centre_tail
    wn += n_node_centre_tail - 1
    end_tails_nodesL[1] = wn - 1
    end_tails_elementsL[1] = we - 1

    beam_number[we:we + n_elem_tail] = 19
    tail_beam_numbersL[1,1] = 19
    x[wn:wn + n_node_tail - 1] = x[end_tails_nodesL[1]]
    y[wn:wn + n_node_tail - 1] = y[end_tails_nodesL[1]] + np.linspace(0.0, span_tail, n_node_tail)[1:]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    elem_stiffness[we:we + n_elem_tail] = 4
    elem_mass[we:we + n_elem_tail] = 4
    boundary_conditions[wn + n_node_tail - 1 -1] = -1
    we += n_elem_tail
    wn += n_node_tail - 1

    beam_number[we:we + n_elem_tail] = 20
    tail_beam_numbersL[1,2] = 20
    x[wn:wn + n_node_tail - 1] = x[end_tails_nodesL[1]]
    y[wn:wn + n_node_tail - 1] = y[end_tails_nodesL[1]] + np.linspace(0.0, -span_tail, n_node_tail)[1:]
    for ielem in range(n_elem_tail):
        conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
        for inode in range(n_node_elem):
            frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    conn[we, 0] = end_tails_nodesL[1]
    elem_stiffness[we:we + n_elem_tail] = 4
    elem_mass[we:we + n_elem_tail] = 4
    boundary_conditions[wn + n_node_tail - 1 -1] = -1
    we += n_elem_tail
    wn += n_node_tail - 1

    # # outer tail 2L
    # beam_number[we:we + n_elem_outer_tail] = 26
    # tail_beam_numbersL[2,0] = 26
    # x[wn:wn + n_node_outer_tail - 1] = np.linspace(0.0, length_outer_tail, n_node_outer_tail)[1:]
    # y[wn:wn + n_node_outer_tail - 1] = y[end_nodesL[2]]
    # for ielem in range(n_elem_outer_tail):
    #     conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
    #     for inode in range(n_node_elem):
    #         frame_of_reference_delta[we + ielem, inode, :] = [0.0, 1.0, 0.0]
    # conn[we, 0] = end_nodesL[2]
    # elem_stiffness[we:we + n_elem_centre_tail] = 0
    # elem_mass[we:we + n_elem_centre_tail] = 0
    # we += n_elem_centre_tail
    # wn += n_node_centre_tail - 1
    # end_tails_nodesL[2] = wn - 1
    # end_tails_elementsL[2] = we - 1
    #
    # beam_number[we:we + n_elem_tail] = 27
    # tail_beam_numbersL[2,1] = 27
    # x[wn:wn + n_node_tail - 1] = x[end_tails_nodesL[2]]
    # y[wn:wn + n_node_tail - 1] = y[end_tails_nodesL[2]] + np.linspace(0.0, span_tail, n_node_tail)[1:]
    # for ielem in range(n_elem_tail):
    #     conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
    #     for inode in range(n_node_elem):
    #         frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]
    # elem_stiffness[we:we + n_elem_tail] = 0
    # elem_mass[we:we + n_elem_tail] = 0
    # boundary_conditions[wn + n_node_tail - 1 -1] = -1
    # we += n_elem_tail
    # wn += n_node_tail - 1
    #
    # beam_number[we:we + n_elem_tail] = 28
    # tail_beam_numbersL[2,2] = 28
    # x[wn:wn + n_node_tail - 1] = x[end_tails_nodesL[2]]
    # y[wn:wn + n_node_tail - 1] = y[end_tails_nodesL[2]] + np.linspace(0.0, -span_tail, n_node_tail)[1:]
    # for ielem in range(n_elem_tail):
    #     conn[we + ielem, :] = ((np.ones((3, ))*(we + ielem)*(n_node_elem - 1)) + np.array([0, 2, 1]))
    #     for inode in range(n_node_elem):
    #         frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]
    # conn[we, 0] = end_tails_nodesL[2]
    # elem_stiffness[we:we + n_elem_tail] = 0
    # elem_mass[we:we + n_elem_tail] = 0
    # boundary_conditions[wn + n_node_tail - 1 -1] = -1
    # we += n_elem_tail
    # wn += n_node_tail - 1

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(x, y)
        plt.show()

    with h5.File(route + '/' + case_name + '.fem.h5', 'a') as h5file:
        coordinates = h5file.create_dataset('coordinates', data=np.column_stack((x, y, z)))
        conectivities = h5file.create_dataset('connectivities', data=conn)
        num_nodes_elem_handle = h5file.create_dataset(
            'num_node_elem', data=n_node_elem)
        num_nodes_handle = h5file.create_dataset(
            'num_node', data=n_node)
        num_elem_handle = h5file.create_dataset(
            'num_elem', data=n_elem)
        stiffness_db_handle = h5file.create_dataset(
            'stiffness_db', data=stiffness_db)
        stiffness_handle = h5file.create_dataset(
            'elem_stiffness', data=elem_stiffness)
        mass_db_handle = h5file.create_dataset(
            'mass_db', data=mass_db)
        mass_handle = h5file.create_dataset(
            'elem_mass', data=elem_mass)
        frame_of_reference_delta_handle = h5file.create_dataset(
            'frame_of_reference_delta', data=frame_of_reference_delta)
        structural_twist_handle = h5file.create_dataset(
            'structural_twist', data=structural_twist)
        bocos_handle = h5file.create_dataset(
            'boundary_conditions', data=boundary_conditions)
        beam_handle = h5file.create_dataset(
            'beam_number', data=beam_number)
        app_forces_handle = h5file.create_dataset(
            'app_forces', data=app_forces)
        lumped_mass_nodes_handle = h5file.create_dataset(
            'lumped_mass_nodes', data=lumped_mass_nodes)
        lumped_mass_handle = h5file.create_dataset(
            'lumped_mass', data=lumped_mass)
        lumped_mass_inertia_handle = h5file.create_dataset(
            'lumped_mass_inertia', data=lumped_mass_inertia)
        lumped_mass_position_handle = h5file.create_dataset(
            'lumped_mass_position', data=lumped_mass_position)

def read_aero_data(filename='inputs/aero_properties.xlsx'):
    import pandas as pd

    xl = pd.ExcelFile(filename)
    sheets = {sheet_name: xl.parse(sheet_name, header=0, index_col=0) for sheet_name in xl.sheet_names}

    aero_data = dict()
    for sheet, val in sheets.items():
        aero_data[sheet] = dict()
        for item in val['value'].items():
            aero_data[sheet][item[0]] = item[1]

    # import pdb; pdb.set_trace()
    return aero_data


def generate_aero_file():
    global x, y, z

    aero_data = read_aero_data()
    print(aero_data)
    # control surfaces
    n_control_surfaces = 0
    control_surface = np.zeros((n_elem, n_node_elem), dtype=int) - 1
    control_surface_type = np.zeros((n_control_surfaces, ), dtype=int)
    control_surface_deflection = np.zeros((n_control_surfaces, ))
    control_surface_chord = np.zeros((n_control_surfaces, ), dtype=int)

    # right wing (surface 0, beams 0, 1, 2, 3)
    type = 'inboard'
    initial_node = 0
    final_node = end_nodesR[-1]
    initial_elem = 0
    final_elem = end_elementsR[-1]
    i_surf = 0
    airfoil_distribution[initial_elem: final_elem + 1, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
    surface_distribution[initial_elem: final_elem + 1] = i_surf
    surface_m[i_surf] = m
    aero_node[initial_node:final_node + 1] = True
    node_counter = 0
    for i_elem in range(initial_elem, final_elem + 1):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            # chord[i_elem, i_local_node] = temp_chord[node_counter]
            chord[i_elem, i_local_node] = aero_data[type]['chord']
            elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
            twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

    # left wing (surface 1, beams 4, 5, 6, 7)
    initial_node = end_nodesR[-1] + 1
    final_node = end_nodesL[-1]
    initial_elem = end_elementsR[-1] + 1
    final_elem = end_elementsL[-1]
    i_surf += 1
    airfoil_distribution[initial_elem: final_elem + 1, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
    surface_distribution[initial_elem: final_elem + 1] = i_surf
    surface_m[i_surf] = m
    aero_node[initial_node:final_node + 1] = True
    node_counter = 0
    for i_elem in range(initial_elem, final_elem + 1):
        for i_local_node in range(n_node_elem):
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = aero_data[type]['chord']
            elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
            twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

    # centre tail
    # import pdb; pdb.set_trace()
    type = 'Ctail'
    elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersC[1]]
    i_surf += 1
    for i_elem in elements:
        for i_node in range(n_node_elem):
            airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
            aero_node[conn[i_elem, i_node]] = True
    surface_distribution[elements] = i_surf
    surface_m[i_surf] = m
    node_counter = 0
    for i_elem in elements:
        for i_local_node in [0, 1, 2]:
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = aero_data[type]['chord']
            elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
            twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

    elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersC[2]]
    i_surf += 1
    for i_elem in elements:
        for i_node in range(n_node_elem):
            airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
            aero_node[conn[i_elem, i_node]] = True
    surface_distribution[elements] = i_surf
    surface_m[i_surf] = m
    node_counter = 0
    for i_elem in elements:
        for i_local_node in [0, 1, 2]:
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = aero_data[type]['chord']
            elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
            twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

    # 0R tail
    type = '0Rtail'
    elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersR[0,1]]
    i_surf += 1
    for i_elem in elements:
        for i_node in range(n_node_elem):
            airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
            aero_node[conn[i_elem, i_node]] = True
    surface_distribution[elements] = i_surf
    surface_m[i_surf] = m
    # temp_chord = np.linspace(tail_chord, tail_chord, final_node + 1 - initial_node + 1)
    node_counter = 0
    for i_elem in elements:
        for i_local_node in [0, 1, 2]:
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = aero_data[type]['chord']
            elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
            twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

    elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersR[0,2]]
    i_surf += 1
    for i_elem in elements:
        for i_node in range(n_node_elem):
            airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
            aero_node[conn[i_elem, i_node]] = True
    surface_distribution[elements] = i_surf
    surface_m[i_surf] = m
    node_counter = 0
    for i_elem in elements:
        for i_local_node in [0, 1, 2]:
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = aero_data[type]['chord']
            elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
            twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

    # 1R tail
    # import pdb; pdb.set_trace()
    type = '0Rtail'
    elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersR[1,1]]
    i_surf += 1
    for i_elem in elements:
        for i_node in range(n_node_elem):
            airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
            aero_node[conn[i_elem, i_node]] = True
    surface_distribution[elements] = i_surf
    surface_m[i_surf] = m
    node_counter = 0
    for i_elem in elements:
        for i_local_node in [0, 1, 2]:
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = aero_data[type]['chord']
            elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
            twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

    elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersR[1,2]]
    i_surf += 1
    for i_elem in elements:
        for i_node in range(n_node_elem):
            airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
            aero_node[conn[i_elem, i_node]] = True
    surface_distribution[elements] = i_surf
    surface_m[i_surf] = m
    node_counter = 0
    for i_elem in elements:
        for i_local_node in [0, 1, 2]:
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = aero_data[type]['chord']
            elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
            twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

    # 0L tail
    type = '0Ltail'
    elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersL[0,1]]
    i_surf += 1
    for i_elem in elements:
        for i_node in range(n_node_elem):
            airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
            aero_node[conn[i_elem, i_node]] = True
    surface_distribution[elements] = i_surf
    surface_m[i_surf] = m
    node_counter = 0
    for i_elem in elements:
        for i_local_node in [0, 1, 2]:
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = aero_data[type]['chord']
            elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
            twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

    elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersL[0,2]]
    i_surf += 1
    for i_elem in elements:
        for i_node in range(n_node_elem):
            airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
            aero_node[conn[i_elem, i_node]] = True
    surface_distribution[elements] = i_surf
    surface_m[i_surf] = m
    node_counter = 0
    for i_elem in elements:
        for i_local_node in [0, 1, 2]:
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = aero_data[type]['chord']
            elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
            twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

    # 1L tail
    # import pdb; pdb.set_trace()
    type = '0Ltail'
    elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersL[1,1]]
    i_surf += 1
    for i_elem in elements:
        for i_node in range(n_node_elem):
            airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
            aero_node[conn[i_elem, i_node]] = True
    surface_distribution[elements] = i_surf
    surface_m[i_surf] = m
    node_counter = 0
    for i_elem in elements:
        for i_local_node in [0, 1, 2]:
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = aero_data[type]['chord']
            elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
            twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

    elements = np.linspace(0, n_elem - 1, n_elem, dtype=int)[beam_number == tail_beam_numbersL[1,2]]
    i_surf += 1
    for i_elem in elements:
        for i_node in range(n_node_elem):
            airfoil_distribution[i_elem, :] = aero_data['airfoil_indices'][aero_data[type]['airfoil']]
            aero_node[conn[i_elem, i_node]] = True
    surface_distribution[elements] = i_surf
    surface_m[i_surf] = m
    node_counter = 0
    for i_elem in elements:
        for i_local_node in [0, 1, 2]:
            if not i_local_node == 0:
                node_counter += 1
            chord[i_elem, i_local_node] = aero_data[type]['chord']
            elastic_axis[i_elem, i_local_node] = aero_data[type]['elastic_axis']
            twist[i_elem, i_local_node] = -aero_data[type]['twist']*np.pi/180

    with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
        airfoils_group = h5file.create_group('airfoils')
        # add one airfoil
        emx07_main = airfoils_group.create_dataset('0', data=load_airfoil('inputs/EMX-07_camber.txt'))
        flat_tail = airfoils_group.create_dataset('1', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))

        # chord
        chord_input = h5file.create_dataset('chord', data=chord)
        dim_attr = chord_input .attrs['units'] = 'm'

        # twist
        twist_input = h5file.create_dataset('twist', data=twist)
        dim_attr = twist_input.attrs['units'] = 'rad'

        # airfoil distribution
        airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)

        surface_distribution_input = h5file.create_dataset('surface_distribution', data=surface_distribution)
        surface_m_input = h5file.create_dataset('surface_m', data=surface_m)
        m_distribution_input = h5file.create_dataset('m_distribution', data=m_distribution.encode('ascii', 'ignore'))

        aero_node_input = h5file.create_dataset('aero_node', data=aero_node)
        elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)

        control_surface_input = h5file.create_dataset('control_surface', data=control_surface)
        control_surface_deflection_input = h5file.create_dataset('control_surface_deflection', data=control_surface_deflection)
        control_surface_chord_input = h5file.create_dataset('control_surface_chord', data=control_surface_chord)
        control_surface_types_input = h5file.create_dataset('control_surface_type', data=control_surface_type)

def load_airfoil(filename):
    data = np.loadtxt(filename, skiprows=1)
    # data[:, 1] *= 10
    # import pdb; pdb.set_trace()
    return data


def generate_naca_camber(M=0, P=0):
    mm = M*1e-2
    p = P*1e-1

    def naca(x, mm, p):
        if x < 1e-6:
            return 0.0
        elif x < p:
            return mm/(p*p)*(2*p*x - x*x)
        elif x > p and x < 1+1e-6:
            return mm/((1-p)*(1-p))*(1 - 2*p + 2*p*x - x*x)

    x_vec = np.linspace(0, 1, 1000)
    y_vec = np.array([naca(x, mm, p) for x in x_vec])
    return x_vec, y_vec


def generate_solver_file():
    file_name = route + '/' + case_name + '.solver.txt'
    settings = dict()
    settings['SHARPy'] = {'case': case_name,
                          'route': route,
                          'flow': flow,
                          'write_screen': 'on',
                          'write_log': 'on',
                          'log_folder': route + '/output/',
                          'log_file': case_name + '.log'}

    settings['BeamLoader'] = {'unsteady': 'on',
                              'orientation': algebra.euler2quat(np.array([0.0,
                                                                          alpha,
                                                                          beta]))}

    settings['AerogridLoader'] = {'unsteady': 'on',
                                  'aligned_grid': 'on',
                                  'mstar': 30,
                                  'freestream_dir': ['1', '0', '0']}

    settings['NonLinearStatic'] = {'print_info': 'on',
                                   'max_iterations': 350,
                                   'num_load_steps': 5,
                                   'delta_curved': 1e-15,
                                   'min_delta': 1e-8,
                                   'gravity_on': gravity,
                                   'gravity': gravity_value}
    settings['StaticUvlm'] = {'print_info': 'on',
                              'horseshoe': 'off',
                              'num_cores': 4,
                              'n_rollup': 0,
                              'rollup_dt': dt,
                              'rollup_aic_refresh': 1,
                              'rollup_tolerance': 1e-4,
                              'velocity_field_generator': 'SteadyVelocityField',
                              'velocity_field_input': {'u_inf': u_inf,
                                                       'u_inf_direction': [1., 0, 0]},
                              'rho': rho}

    settings['StaticCoupled'] = {'print_info': 'on',
                                 'structural_solver': 'NonLinearStatic',
                                 'structural_solver_settings': settings['NonLinearStatic'],
                                 'aero_solver': 'StaticUvlm',
                                 'aero_solver_settings': settings['StaticUvlm'],
                                 'max_iter': 100,
                                 'n_load_steps': n_step,
                                 'tolerance': tolerance,
                                 'relaxation_factor': relaxation_factor}

    settings['BeamPlot'] = {'folder': route + '/output/',
                            'include_rbm': 'on',
                            'include_applied_forces': 'on',
                            'include_forward_motion': 'off'}

    settings['AerogridPlot'] = {'folder': route + '/output/',
                                'include_rbm': 'on',
                                'include_forward_motion': 'off',
                                'include_applied_forces': 'on',
                                'minus_m_star': 0,
                                'u_inf': u_inf,
                                'dt': dt}
    settings['BeamLoads'] = dict()

    import configobj
    config = configobj.ConfigObj()
    config.filename = file_name
    for k, v in settings.items():
        config[k] = v
    config.write()



clean_test_files()
generate_fem()
generate_aero_file()
generate_solver_file()
