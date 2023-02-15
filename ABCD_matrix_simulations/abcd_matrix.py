import numpy as np
from numpy.random import default_rng
rng = default_rng() 

import re
from tqdm.notebook import tqdm
import multiprocessing

import schemdraw
import schemdraw.elements as elm


# library
class MatrixLibrary():

    def __init__(self):
        self.library = True

    ## Definition of the components' matrices
    # Matrices of the elements listed in the previous paragraph, and using 
    # the parameters dictionary at the beginning of the notebook. They
    # will be multiplied together to obtain the final spectrum of the device.


    # input/output resistance
    def Rb(omega, parameters): 
        rb = parameters['line_resistance']

        rb_matrix = np.array( [ [1, 1/rb],
                                [0, 1   ] ], dtype=object)
        return rb_matrix


    # resistance to ground    
    def Rg(omega, parameters): 
        rg = parameters['ground_resistance']

        rg_matrix = np.array( [ [1   , 0],
                                [1/rg, 1] ], dtype=object)
        return rg_matrix


    # capacitance to ground    
    def Cg(omega, parameters): 
        cg = parameters['array_junction_ground_capacitance']

        cg_matrix = np.array( [ [1          , 0],
                                [1j*omega*cg, 1] ], dtype=object)
        return cg_matrix


    # parallel RLC in series for single junctions
    def JJ(omega, parameters):
        lj = parameters['array_junction_inductance']
        cj = parameters['array_junction_capacitance']
        rj = parameters['array_junction_parallel_resistance']

        rlcj = 1j*omega*cj + 1/(1j*omega*lj) + 1/rj

        jj_matrix = np.array( [ [1, 1/rlcj], 
                                [0, 1     ] ], dtype=object)
        return jj_matrix



    # junction and ground capacitance
    def J(omega, parameters):
        # calling the class withing itself is not the best idea, it could be done with "self" 
        # but then every component needs as an input the library as well, quite useless.

        junction_matrix = MatrixLibrary.JJ(omega, parameters)
        ground_capacitance_matrix = MatrixLibrary.Cg(omega, parameters)

        j_matrix = np.matmul(junction_matrix, ground_capacitance_matrix)

        return j_matrix


    # UJJ, which to be honest is also a parallel RLC circuit
    def UJ(omega, parameters): 
        lu = parameters['ultrasmall_junction_inductance']
        cu = parameters['ultrasmall_junction_capacitance']
        ru = parameters['ultrasmall_junction_resistance']

        rlcu = 1j*omega*cu + 1/ru + 1/(1j*omega*lu)

        uj_matrix = np.array( [ [1, 1/rlcu],
                                [0, 1     ] ], dtype=object)
        return uj_matrix


    # SSH period
    def SSH(omega, parameters):
        # to make an SSH period, the large junctions are the ones normally defined, while the small
        # ones are build with a temporary parameter dictionary which scales these numbers of the amount
        # defined by "ssh_junctions_ratio".

        # build the large junction
        large_ssh_junction = MatrixLibrary.J(omega, parameters)

        # build the small junction
        ssh_temp_parameters = parameters.copy()

        ssh_temp_parameters.update({'array_junction_inductance': parameters['array_junction_inductance'] * parameters['ssh_junctions_ratio']})
        ssh_temp_parameters.update({'array_junction_capacitance': parameters['array_junction_capacitance'] / parameters['ssh_junctions_ratio']})
        ssh_temp_parameters.update({'array_junction_parallel_resistance': parameters['array_junction_parallel_resistance'] * parameters['ssh_junctions_ratio']})
        ssh_temp_parameters.update({'array_junction_ground_capacitance': parameters['array_junction_ground_capacitance'] / parameters['ssh_junctions_ratio']})

        small_ssh_junction = MatrixLibrary.J(omega, ssh_temp_parameters)

        # calculating the matrix
        ssh_matrix = np.identity(2)
        for i in range(parameters['ssh_period_N']):
            ssh_matrix = np.matmul(ssh_matrix, large_ssh_junction)

        for i in range(parameters['ssh_period_n']):
            ssh_matrix = np.matmul(ssh_matrix, small_ssh_junction)

        return ssh_matrix


    # resistance in series
    def Rs(omega, parameters): 
        rs = parameters['series_resistance']

        rs_matrix = np.array( [ [1, 1/rs],
                                [0, 1   ] ], dtype=object)
        return rs_matrix


    # capacitance in series
    def Cs(omega, parameters): 
        cs = parameters['series_capacitance']

        cs_matrix = np.array( [ [1, 1j*omega*cs],
                                [0, 1          ] ], dtype=object)
        return cs_matrix


    # short to ground
    def Sg(omega, parameters):
        sg = 1e-21
        sg_matrix = np.array( [ [1,     0],
                                [1/sg, 1] ])
        return sg_matrix


    # open to ground
    def Og(omega, parameters):
        og = 1e21
        og_matrix = np.array( [ [1,      0],
                                [1/og,   1] ])
        return og_matrix


# methods
class ABCDMethods():

    # Calculate the ABCD matrix starting from a string which defines the circuit, and the corresponding parameters
    def array_abcd(omega, circuit, parameters):
    
        # start with the identity matrix
        ABCD = np.identity(2)
        
        # add the circuit components
        circuit_parts = circuit.split()
        
        for part in tqdm( circuit_parts, leave=False ):
        #for part in circuit_parts:
            numeric_part = re.findall('\d+', part)[0]          # find the number of components from the name
            component_part = re.findall('[a-zA-Z]+', part)[0]  # find the type of component from the name
            
            # call the function in MatrixLibrary corresponding to the component, if present
            if hasattr(MatrixLibrary, component_part) == True:
                component = getattr(MatrixLibrary, component_part)
            else: 
                raise NameError('There is no component named '+component_part+'.')

            # generate a random array, to be used in case disorder is on
            disorder = parameters['disorder_sigma'] * rng.standard_normal( int(numeric_part) )

            for i in range( int(numeric_part) ):

                if parameters['add_disorder'] == True:
                    # if disorder is on, one parameter is spread out
                    disordered_parameter = parameters['disorder_parameter']

                    disordered_parameters = parameters.copy()
                    disordered_parameters.update({disordered_parameter: parameters[disordered_parameter] + disorder[i]*parameters[disordered_parameter]})

                    ABCD = np.matmul(ABCD, component(omega, disordered_parameters))

                else:
                    # do the multiplication, maybe we should parallelise this
                    ABCD = np.matmul(ABCD, component(omega, parameters))

        return ABCD



    ## S-coefficients

    # reflection coefficients
    def s11(A, parameters, dB_out=False):

        # read the desired impedance
        z0 = parameters['line_impedance']

        # calculate the s11 spectrum (complex)
        s11 = ( A[0][0] + A[0][1]/z0 - A[1][0]*z0 - A[1][1] ) / ( A[0][0] + A[0][1]/z0 + A[1][0]*z0 + A[1][1] )

        if dB_out == True:
            s11 = 20*np.log10( np.abs(s11) )

        return s11


    def s22(A, parameters, dB_out=False):

        # read the desired impedance
        z0 = parameters['line_impedance']

        # calculate the s11 spectrum (complex)
        s22 = ( - A[0][0] + A[0][1]/z0 - A[1][0]*z0 + A[1][1] ) / ( A[0][0] + A[0][1]/z0 + A[1][0]*z0 + A[1][1] )

        if dB_out == True:
            s22 = 20*np.log10( np.abs(s22) )

        return s22



    # transmission coefficients
    def s12(A, parameters, dB_out=False):

        # read the desired impedance
        z0 = parameters['line_impedance']

        # calculate the s11 spectrum (complex)
        s12 = 2 * ( A[0][0]*A[1][1]  -  A[0][1]*A[1][0] ) / ( A[0][0] + A[0][1]/z0 + A[1][0]*z0 + A[1][1] )

        if dB_out == True:
            s12 = 20*np.log10( np.abs(s12) )

        return s12


    def s21(A, parameters, dB_out=False):

        # read the desired impedance
        z0 = parameters['line_impedance']

        # calculate the s21 spectrum (complex)
        s21 = 2 / ( A[0][0] + A[0][1]/z0 + A[1][0]*z0 + A[1][1] )

        if dB_out == True:
            s21 = 20*np.log10( np.abs(s21) )

        return s21


# utilities
class ABCDUtils():

    # class used for some handy functions that can help

    # useful variables
    def __init__(self):
        self.numeric_part = 'Not defined.'


    def show_array_properties(parameters):
        # print some useful properties of the device
        hbar = 1.05e-34
        e = 1.60e-19

        Lj = parameters['array_junction_inductance']
        Cj = parameters['array_junction_capacitance']
        Cg = parameters['array_junction_ground_capacitance']


        print('Plasma frequency =', np.round( 1/(2*np.pi) / np.sqrt(Lj*Cj) / 1e9, 3), 'GHz')
        print('Critical current =', np.round(  1e9 * hbar / 2 / e / Lj, 3), 'nA')
        print('Line impedance =', np.round( np.sqrt(Lj / Cg) / 1e3, 3), 'kOhm')



    def draw_circuit(circuit, save_scheme = False):

        ## Let's draw some circuits
        # It can be useful (and cool) to see the circuit that is being considered.
        # This function draws the circuit starting from the same input used for the simulation.

        ## First define the various parts

        # input line
        with schemdraw.Drawing(show=False) as input_d:
            input_d += elm.Line().right().length(1.75).idot(open=True).dot().label('1', loc='left')
            input_d.push()
            input_d += elm.Line().left().length(0.75)
            input_d += elm.ResistorIEC().down().length(2).label('$Z_0$', halign='center', valign='center', rotate=True, ofst=-0.25)
            input_d += elm.Line().left().length(1).dot(open=True)
            input_d += elm.Line().right().length(1.75).dot()
            input_d.pop()
            
        # output line
        with schemdraw.Drawing(show=False) as output_d:
            output_d += elm.Line().right().length(1.75).idot().dot(open=True).label('2', loc='right')
            output_d += elm.Line().left().length(1)
            output_d += elm.ResistorIEC().down().length(2).label('$Z_0$', halign='center', valign='center', rotate=True, ofst=-0.25)
            output_d += elm.Line().left().length(0.75).dot()
            output_d += elm.Line().right().length(1.75).dot(open=True)


        # series resistance 
        with schemdraw.Drawing(show=False) as series_r:
            series_r += elm.Resistor().scale(0.75).right().length(2).idot()
            series_r.push()
            series_r += elm.Line().down().length(2).color('white')
            series_r += elm.Line().left().length(2).dot()
            series_r.pop()

        # series resistance 2, because it is necessary (and I'm lazy)
        with schemdraw.Drawing(show=False) as series_r2:
            series_r2 += elm.Resistor().scale(0.75).right().length(2).idot()
            series_r2.push()
            series_r2 += elm.Line().down().length(2).color('white')
            series_r2 += elm.Line().left().length(2).dot()
            series_r2.pop()

        # series capacitance 
        with schemdraw.Drawing(show=False) as series_c:
            series_c += elm.Capacitor().right().length(2).idot()
            series_c.push()
            series_c += elm.Line().down().length(2).color('white')
            series_c += elm.Line().left().length(2).dot()
            series_c.pop()

        # capacitance to ground
        with schemdraw.Drawing(show=False) as ground_c:
            ground_c.push()
            ground_c += elm.Capacitor().down().length(2).dot().idot()
            ground_c.pop()

        # resistance to ground
        with schemdraw.Drawing(show=False) as ground_r:
            ground_r += elm.Line().right().length(0.75)
            ground_r.push()
            ground_r += elm.Resistor().down().length(2).dot().idot()
            ground_r += elm.Line().left().length(0.75).idot()
            ground_r += elm.Line().right().length(1.5)
            ground_r.pop()
            ground_r += elm.Line().right().length(0.75)

        # short to ground
        with schemdraw.Drawing(show=False) as ground_short:
            ground_short += elm.Line().right().length(0.5)
            ground_short.push()
            ground_short += elm.Line().down().length(2).dot().idot()
            ground_short += elm.Line().left().length(0.5)
            ground_short.pop()

        # open to ground
        with schemdraw.Drawing(show=False) as ground_open:
            ground_open += elm.Line().right().length(1)
            ground_open.push()
            ground_open += elm.ResistorIEC().down().label('$\infty$', halign='center', valign='center', rotate=True, ofst=-0.25).length(2).dot().idot()
            ground_open += elm.Line().left().length(1)
            ground_open.pop()

        # josephson junction transmission line    
        with schemdraw.Drawing(show=False) as jj_line:
            jj_color = 'slateblue'
            jj_line += elm.Line().length(0.5)
            jj_line += elm.Line().length(0.5).up().color(jj_color)
            jj_line += elm.Resistor().scale(0.5).length(1).right().color(jj_color)
            jj_line += elm.Line().length(1).down().color(jj_color)
            jj_line += elm.Capacitor().scale(0.75).length(1).left().color(jj_color)
            jj_line += elm.Line().length(0.5).up().color(jj_color)
            jj_line += elm.Inductor().scale(0.75).length(1).right().color(jj_color)
            jj_line += elm.Line().length(0.5).right()

            jj_line.push()
            jj_line += elm.Capacitor().down().length(2).dot().idot()
            jj_line += elm.Line().length(2).left()
            jj_line.pop()


        # ultrasmall junction, which actually is an RLC, but it's fun to cheat        
        with schemdraw.Drawing(show=False) as ujj:
            ujj_color = 'darkred'
            ujj += elm.Line().length(0.5)
            ujj += elm.Line().length(0.5).up().color(ujj_color)
            ujj += elm.Resistor().scale(0.5).length(1).right().color(ujj_color)
            ujj += elm.Line().length(1).down().color(ujj_color)
            ujj += elm.Capacitor().scale(0.75).length(1).left().color(ujj_color)
            ujj += elm.Line().length(0.5).up().color(ujj_color)
            ujj += elm.Josephson().scale(0.75).length(1).right().color(ujj_color)
            ujj += elm.Line().length(0.5).right()
            ujj.push()
            ujj += elm.Line().length(2).down().color('white')
            ujj += elm.Line().length(2).left()
            ujj.pop()


        # scaled version of the junction transmission line (used to draw ssh)
        with schemdraw.Drawing(show=False) as jj_line_scaled:
            jj_color = 'slateblue'
            scale = 0.5

            jj_line_scaled += elm.Line().length(scale*0.5)
            jj_line_scaled += elm.Line().length(scale*0.5).up().color(jj_color)
            jj_line_scaled += elm.Resistor().scale(scale*0.5).length(scale*1).right().color(jj_color)
            jj_line_scaled += elm.Line().length(scale*1).down().color(jj_color)
            jj_line_scaled += elm.Capacitor().scale(scale*0.75).length(scale*1).left().color(jj_color)
            jj_line_scaled += elm.Line().length(scale*0.5).up().color(jj_color)
            jj_line_scaled += elm.Inductor().scale(scale*0.75).length(scale*1).right().color(jj_color)
            jj_line_scaled += elm.Line().length(scale*0.5).right()

            jj_line_scaled.push()
            jj_line_scaled += elm.Capacitor().down().length(2).dot().idot().scale(scale)
            jj_line_scaled += elm.Line().length(scale*2).left()
            jj_line_scaled.pop()


        # ssh line (can be improved)
        with schemdraw.Drawing(show=False) as ssh:
            ssh_color = 'darkolivegreen'

            ssh += elm.ElementDrawing(jj_line).color(ssh_color)
            ssh += elm.ElementDrawing(jj_line_scaled).color(ssh_color)



        # when there's a lot of them    
        with schemdraw.Drawing(show=False) as a_lot_later:   
            a_lot_later += elm.DotDotDot().scale(0.5)#.label( str(self.numeric_part) )
            a_lot_later.push()
            a_lot_later += elm.Line().length(2).down().color('white')
            a_lot_later += elm.DotDotDot().scale(0.5).left()
            a_lot_later.pop()       


        # it is important that there is a one-to-one correspondence between these two following lists
        scheme_library = [ series_r,   ground_c, ground_r,   jj_line,    ujj,     ssh,    series_r2, series_c, ground_open, ground_short ]
        elements_names = [ 'Rb',       'Cg',     'Rg',       'J',        'UJ',    'SSH',  'Rs',      'Cs'    , 'Og'       , 'Sg'         ]
        

        ## Draw the circuit components
        # This part follows the same procedure used to multiply the matrices

        # sort the pieces
        circuit_parts = circuit.split()

        # actual draw
        with schemdraw.Drawing() as c:
            
            # input feedline
            c += elm.ElementDrawing(input_d)

            # sort the various parts of the circuit
            for part in circuit_parts:
                numeric_part = re.findall('\d+', part)[0]                      # find the number of components from the name
                component_part = re.findall('[a-zA-Z]+', part)[0]              # find the type of component from the name
                                
                # raise an error/warning if an element is not present
                if component_part not in elements_names:
                    raise NameError('The circuit part called '+component_part+' is not present in the schemes library.')
                else:
                    # find the element index corresponding to the name
                    index = elements_names.index(component_part)                   

                


                # choose whether to put all the components or the dots
                if int(numeric_part) <= 3:
                    for i in range( int(numeric_part) ):
                        c += elm.ElementDrawing(scheme_library[index])         # add the element to the drawing
                else:
                    c += elm.ElementDrawing(scheme_library[index])             # add element dots... element
                    c += elm.ElementDrawing(a_lot_later).label(numeric_part, fontsize=16)
                    c += elm.ElementDrawing(scheme_library[index])
            
            # output feedline
            c += elm.ElementDrawing(output_d)

            if save_scheme == True:
                c.save('circuit.pdf', dpi=300)


        return







