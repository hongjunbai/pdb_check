#!/usr/bin/env python2.7
""" Read and Parse PDB files 

Author: Hongjun Bai  (bbhhjj@gmail.com)

Last modified: 06/03/2014, by HB
Fixed all known problems before 06/03/2014

TODO: update to python3
"""    

import sys
import math
import argparse
import gzip
import itertools as it
import copy


""" 20 amino acids:
    ALA     A    alanine   
    CYS     C    cysteine
    ASP     D    aspartic acid
    GLU     E    glutamic acid
    PHE     F    phenylalanine
    GLY     G    glycine
    HIS     H    histidine
    ILE     I    isoleucine
    LYS     K    lysine
    LEU     L    leucine
    MET     M    methionine
    ASN     N    asparagine
    PRO     P    proline
    GLN     Q    glutamine
    ARG     R    arginine
    SER     S    serine
    THR     T    threonine
    VAL     V    valine
    TRP     W    tryptophan
    TYR     Y    tyrosine
    MSE     M    Se-Met
"""

AA1to3 = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
          'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
          'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
          'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}
AA3to1 = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
          'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
          'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
          'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
          'MSE': 'M', 'UNK': 'X'}
BackBoneAtom = set(['C', 'O', 'N', 'CA'])

# Useful functions for structure process
def map_ab(a_prot, b_prot, atom_name):
    """ Get index map {a_prot:b_prot}

    Keyword arguments:
    a_prot -- pdbps.Protein instance
    b_prot -- pdbps.Protein instance
    atom_name -- atom_name of selected atom in PDB

    Output:
    {a_prot_resid: b_prot_resid, ...}

    """
    shift = 0  # assuming SeqID is identical in a_prot and b_prot 
    aidx_2_seqid = dict((i, res.name+str(res.id)) for i, res in enumerate(a_prot.residues))
    seqid_2_bidx = dict((res.name+str(res.id+shift), i) for i, res in enumerate(b_prot.residues))
    map = {}
    for a_idx, seqid in aidx_2_seqid.items():
        try:
            b_idx = seqid_2_bidx[seqid]
            atom = b_prot.residues[b_idx].atom[atom_name]
        except KeyError:  # Either residue not in b_prot or selected atom not in atomDict
            continue
        else:
            map[a_idx] = b_idx
    return map

def sqdist(res0, res1, atom_name0, atom_name1):
    try:
        coor0 = res0.atom[atom_name0].coor
        coor1 = res1.atom[atom_name1].coor
    except ValueError:
        sys.stderr.write("%s of %s and/or %s of %s is absent" %(atom_name0, res0.name+res0.chain_id+str(res0.id), atom_name1, res1.name+res1.chain_id+str(res1.id)))
    else:
        dist2 = sum((a-b)*(a-b) for a, b in it.izip (coor0, coor1))
    return dist2

try:
    import numpy
except ImportError:
    print("NumPy is used for distance calcualtion. Please install NumPy first")
    print("if you plan to use sqdist_mat() and sele_by_dist().")
else:
    def sqdist_mat(xyza, xyzb):
        ''' Get the distance matrix between coords array xyza and xyzb.

        Input: 
            xyza: [[xa1, ya1, za1], [xa2, ya2, za2], ...]
            xyzb: [[xb1, yb1, zb1], [xb2, yb2, zb2], ...]

        Output:
            distmatrix: (an x bn)
            [[D_a1_b1, D_a1_b2, D_a1_b3, ..., D_a1_bn], 
             [D_a2_b1, D_a2_b2, D_a2_b3, ..., D_a2_bn], 
             .
             .
             .
             [D_an_b1, D_an_b2, D_an_b3, ..., D_an_bn], 
        '''
        sizea = xyza.shape[0]
        sizeb = xyzb.shape[0]
        mat_a = xyza.reshape(sizea, 1, 3)
        mat_a = mat_a.repeat(sizeb, axis=1)
        # mat_a:
        # [[[xa1, ya1, za1], [[xa1, ya1, za1], ...],
        #  [[xa2, ya2, za2], [[xa2, ya2, za2], ...], 
        #  .
        #  .
        #  .
        #  [[xan, yan, zan], [[xan, yan, zan], ...]]
        mat_b = xyzb.reshape(1, sizeb, 3)
        mat_b = mat_b.repeat(sizea, axis=0)
        # mat_b:
        # [[[xb1, yb1, zb1], [xb2, yb2, zb2], ...],
        #  [[xb1, yb1, zb1], [xb2, yb2, zb2], ...],
        #  .
        #  .
        #  .
        #  [[xb1, yb1, zb1], [xb2, yb2, zb2], ...]]
        dist = mat_a - mat_b
        dist = numpy.sum(dist * dist, axis=2)
        return dist

    def sele_by_dist(atoms0, atoms1, dist_cutoff):
        coords0 = numpy.array([a.coor for a in atoms0])
        coords1 = numpy.array([a.coor for a in atoms1])
        dist = sqdist_mat(coords0, coords1)
        min_dist = numpy.min(dist, axis=1)
        selected = (min_dist < dist_cutoff*dist_cutoff)
        selected_atoms = [atom for inside, atom in it.izip(selected, atoms0) if inside]
        return selected_atoms 
#

if sys.version_info < (3,):
    range = xrange

AllowedHetRes = set('MSE NAG MAN BMA'.split())

class Protein:
    """  Parse PDB file and provide a basic framework to work with protein structure.

    Example:
    prot = Protein('mypdbfile.pdb')  # or prot = Protein('mypdbfile.pdb.gz')
    xyz = [a.coor for a in prot.atoms()]  # or xyz = [a.coor for a in prot.atoms('CA')]
    print('Contained chains: %s' %(''.join(prot.chains())))
    print('%s' % (prot.seq()))
    newBFs = [1]*prot.size()
    prot.assign_bf(newBFs)
    prot.resort()
    prot.fwrite('processed.pdb')  # or prot.fwrite('processed.pdb.gz')

    """
    def atoms(self, atom_name=None, pos0=None, pos1=None, chains=None):
        """ Shorthand to access atoms (a generator) """
        if pos0 is None or pos0 < 0 or pos0 > pos1:
            i = 0
        else:
            i = pos0
        if pos1 is None or pos1 > self.size() or i >= pos1:
            pos1 = self.size()
        j = 0
        if atom_name is None:
            while i < pos1:
                res = self.residues[i]
                if chains is None or res.chain_id in chains:
                    yield res.atoms[j]
                    if j < res.size() - 1:
                        j += 1
                    else:
                        i += 1; j = 0
                else:
                    i += 1
        else:  # Limite atom_name in ['C', 'N', 'O', 'CA', 'CB']
            assert((atom_name in BackBoneAtom) or atom_name == 'CB')
            while i < pos1:
                res = self.residues[i]
                if chains is None or res.chain_id in chains:
                    try:
                        if atom_name == 'CB' and res.name == 'GLY':
                            yield res.atom['CA']
                        else:
                            yield res.atom[atom_name]
                    except KeyError:
                        print('(%s) residue %s:%s %3d has no %s atom.'
                              %(self.name, res.chain_id, res.name, res.id, atom_name))
                i += 1

    def size(self):
        return len(self.residues)

    def seq(self, chains=None):
        myseq = []
        for res in self.residues:
            if ((chains is None or res.chain_id in chains)):
                if res.name in AA3to1:
                    myseq.append(AA3to1[res.name])
                else:
                    myseq.append('X')
        return ''.join(myseq)

    def chains(self):
        return set(res.chain_id for res in self.residues)

    def assign_bf(self, bf, scale=True):
        assert(len(self.residues) == len(bf))
        if scale:
            upper = max(bf)
            lower = min(bf)
            if lower == upper: lower -= 1
            assigned = [(x-lower)*98.0/(upper-lower)+1.0  for x in bf]
        else:
            assigned = bf
        for res, property in it.izip(self.residues, assigned):
            for atom in res.atoms:
                atom.bf = property

    def fwrite(self, name, msg=None):
        with open_loudly(name, 'w', msg) as outfile:
            outfile.write(str(self))

    def resort(self, first_rid=1, first_aid=1):
        for i, residue in enumerate(self.residues):
            residue.id = first_rid + i
        for i, atom in enumerate(self.atoms()):
            atom.serial = first_aid + i

    def __init__(self, source=None, atom_name=None, chains=None):
        pdblines = [] 
        if isinstance(source, str):     #
            with open_loudly(source) as infile:
                pdblines = infile.readlines()
            self.name = source
        elif isinstance(source, file):  #
        #elif has_method(source, "readlines"):  #
            source.seek(0)
            pdblines = source.readlines()
            self.name = source.name
        elif isinstance(source, list) and isinstance(source[0], Residue): #
            self.name = 'None'
            self.residues = []
            for res in source:
                if (((chains is not None) and (res.chain_id not in chains)) or
                    ((atom_name is not None) and (atom_name not in res.atom))):
                    continue
                if atom_name is not None:
                    temp_res.atoms = [res.atom[atom_name]]
                    temp_res.atom = {atom_name: res.atom[atom_name]}
                else:
                    temp_res = copy.deepcopy(res)
                self.residues.append(temp_res)
        else:
            pass  # Not decided whether I need other type of constructor yet
        if pdblines:
            pdbatoms = []
            for line in pdblines:
                # Atom record of protein (including MSE, which may start with HETATOM)
                if ((line.startswith('ATOM') or  # atom line
                    (line.startswith('HETATM') and line[17:20] in AllowedHetRes)) and  # MSE or Glycans
                    (atom_name is None or line[12:16].strip() == atom_name) and  # atom name
                    (chains is None or line[21] in chains)):  # chain or chains
                    pdbatoms.append(parse_atom(line))
                if line.startswith('END'): break  # Keep only the first model
            assert(pdbatoms)
            self.residues = []
            raw_res = []
            for i, atom in enumerate(pdbatoms[:-1]):
                raw_res.append(atom)
                if atom.resid != pdbatoms[i+1].resid or atom.icode != pdbatoms[i+1].icode:
                    self.residues.append(Residue(raw_res))
                    raw_res = []
            raw_res.append(pdbatoms[-1])
            self.residues.append(Residue(raw_res))
        if self.residues:
            self.residue = dict((res.uniq_id(), res) for res in self.residues)

    def __str__(self):
        header = "REMARK This pdb file is generated by pdblib.py\n"
        header += "REMARK Input file: "+self.name+'\n'
        header += "REMARK Included chains: "+' '.join(self.chains())+'\n'
        if self.residues:
            ch0 = self.residues[0].chain_id
            body = []
            for i, res in enumerate(self.residues):
                if res.chain_id != ch0:  # Chain TER
                    last_res = self.residues[i-1]
                    last_atom = last_res.atoms[-1]
                    body.append('%-6s%5d      %3s%2s%4d\n' %('TER', last_atom.serial+1, last_res.name, last_res.chain_id, last_res.id))
                    ch0 = res.chain_id
                body.append(str(res))
            # TER of last chain
            last_res = self.residues[-1]
            last_atom = last_res.atoms[-1]
            body.append('%-6s%5d      %3s%2s%4d\n' %('TER', last_atom.serial+1, last_res.name, last_res.chain_id, last_res.id))
            body = ''.join(record for record in body)
        else:
            body = "REMARK No residues found!\n"
        tail = 'END\n'
        return header + body + tail

class Residue:
    def uniq_id(self):
        id = self.chain_id + self.name + str(self.id) + self.icode
        return id.strip()

    def size(self):
        return len(self.atoms)

    def to_ala(self):
        if self.name is not 'G':  # Gly is kept as is.
            ala_atoms = set(['C', 'O', 'N', 'CA', 'CB'])
            self.atoms = [a for a in self.atoms if a.name in ala_atoms]
            self.name = 'A'
            self.atom = dict((x.name, x) for x in self.atoms)

    def __init__(self, pdbatoms):
        try:
            atom0 = pdbatoms[0]
        except KeyError:
            sys.stderr.write("Empty pdbatom list used for residue initialization")
            sys.exit()
        else:
            self.name = atom0.resname
            self.chain_id = atom0.chain_id
            self.id = atom0.resid
            self.icode = atom0.icode
            self.seg = atom0.seg
            #
            self.atoms = [Atom(pdbatom, self) for pdbatom in pdbatoms]
            self.atom = dict((x.name, x) for x in self.atoms)

    def __str__(self):
        return ''.join(str(atom) for atom in self.atoms)

class Atom:
    def __init__(self, pdbatom, parent=None):
        self.type = pdbatom.type
        self.serial = pdbatom.serial
        self.name = pdbatom.name
        self.altloc = pdbatom.altloc
        self.coor = pdbatom.coor
        self.occupancy = pdbatom.occupancy
        self.bf = pdbatom.bf
        self.element = pdbatom.element
        self.charge = pdbatom.charge
        #
        self.parent = parent
        
    def __str__(self):
        res = self.parent
        if self.name[0] in '0123456789':
            name_format = '%-4s'
        elif len(self.name) > 3:
            name_format  = '%4s'
        else:
            name_format = ' %-3s'
        atom_format = '%-6s%5d '+name_format+'%1s%3s%2s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s%2s\n'
        return atom_format %(self.type, self.serial, self.name, self.altloc, res.name,
                res.chain_id, res.id, res.icode, self.coor[0], self.coor[1], self.coor[2],
                self.occupancy, self.bf, res.seg, self.element, self.charge)
                
def parse_atom(line):
    """
    #  Atom record format in pdbfile (downloaded from rcsb.org, 08-22, 2006)
    #+      COLUMNS   DATA TYPE       FIELD         DEFINITION                            
    #+ -----------------------------------------------------------------------------
    #+ 0    1 -  6    Record name     ATOM                                            
    #+ 1    7 - 11    Integer         serial        Atom serial number.                   
    #+ 2    13 - 16   Atom            name          Atom name.                            
    #+ 3    17        Character       altLoc        Alternate location indicator.         
    #+ 4    18 - 20   Residue name    resName       Residue name.                         
    #+ 5    22        Character       chainID       Chain identifier.                     
    #+ 6    23 - 26   Integer         resSeq        Residue sequence number.              
    #+ 7    27        AChar           iCode         Code for insertion of residues.       
    #+ 8    31 - 38   Real(8.3)       x             Orthogonal coordinates for X in Angstroms
    #+ 9    39 - 46   Real(8.3)       y             Orthogonal coordinates for Y in Angstroms
    #+ 0    47 - 54   Real(8.3)       z             Orthogonal coordinates for Z in Angstroms
    #+ 1    55 - 60   Real(6.2)       occupancy     Occupancy.                            
    #+ 2    61 - 66   Real(6.2)       tempFactor    Temperature factor.                   
    #+ 3    73 - 76   LString(4)      segID         Segment identifier, left-justified.   
    #+ 4    77 - 78   LString(2)      element       Element symbol, right-justified.      
    #+ 5    79 - 80   LString(2)      charge        Charge on the atom.
    """
    a = line.rstrip()
    assert (len(a) >= 54)
    class PdbAtomEntry: pass
    pdbatom = PdbAtomEntry()
    pdbatom.type = a[0:6].rstrip()
    pdbatom.serial = int(a[6:11])
    pdbatom.name = a[11:16].strip()
    pdbatom.altloc = a[16]
    pdbatom.resname = a[17:20]
    pdbatom.chain_id = a[21]
    pdbatom.resid = int(a[22:26])
    pdbatom.icode = a[26]
    try:
        coor = [float(a[30:38]), float(a[38:46]), float(a[46:54])]
    except ValueError:  # Handle it if coords is missing
        coor = [0.0, 0.0, 0.0]
    # "try ... except ..." approach is faster than following way:
    # coor = [0.0, 0.0, 0.0]
    # if a[27:38].strip() != '': coor[0] = float(a[27:38])
    # if a[38:46].strip() != '': coor[1] = float(a[38:46])
    # if a[46:54].strip() != '': coor[2] = float(a[46:54])
    pdbatom.coor = coor
    #In some CASP models, the format of following parts is wrong or missing.
    try:
        pdbatom.occupancy = float(a[54:60])
    except ValueError:
        pdbatom.occupancy = 1.0
    try:
        pdbatom.bf = float(a[60:66])
    except ValueError:
        pdbatom.bf = 0.0
    try: 
        pdbatom.seg = a[72:76].strip()
    except ValueError:
        pdbatom.seg = ' '
    try:
        pdbatom.element = a[76:78].strip()
    except ValueError:
        pdbatom.element = ' '
    try:
        pdbatom.charge = a[78:80].strip()
    except ValueError:
        pdbatom.charge = ' '
    return pdbatom

# Detect the defects in structure
#Auxiliary functions of check_stru():
def find_altloc(protein):
    altloc_res = []
    for res in protein.residues:
        altloc_tags = list(set(a.altloc for a in res.atoms if a.altloc != ' '))
        altloc_tags.sort()  # Ordered like: ['A', 'B', 'C']
        if len(altloc_tags) > 1:
            altloc_res.append([res, altloc_tags])
    return altloc_res

def find_gap(protein, cutoff=3.0):
    gaps = []
    for i, res in enumerate(protein.residues[:-1]):
        next_res = protein.residues[i+1]
        if (res.chain_id == next_res.chain_id and 
            (res.id + 1 != next_res.id or res.icode != next_res.icode)): 
            try:
                distij = sqdist(res, next_res, 'C', 'N')
            except KeyError:
                distij = 1.0E8
            if distij > cutoff * cutoff:
                gaps.append([res, next_res, math.sqrt(distij)])
    return gaps

def clean_hydrogen(protein, loudly=False):
    def hydrogen(a):
        return a.name.startswith('H') or ((a.name[0] in '0123456789') and a.name[1] == 'H')
    if loudly: cnt = 0
    for res in protein.residues:
        if loudly: cnt += len(res.atoms)
        res.atoms = [atom for atom in it.ifilterfalse(hydrogen, res.atoms)]
        if loudly: cnt -= len(res.atoms)
    if loudly and cnt > 0:
        print("%d hydrogen atoms have been removed." %(cnt))

def clean_missing_ca(protein, loudly=False):
    if loudly:
        for res in protein.residues:
            if 'CA' not in res.atom and res.name in AA3to1:
                print("Residue %s:%s%c %3d has no CA atom, deleted." 
                        %(res.chain_id, res.name, res.icode, res.id))
    protein.residues = filter(lambda res: 'CA' in res.atom or res.name in AllowedHetRes, protein.residues)
    if protein.residues:
        protein.residue = dict((res.uniq_id(), res) for res in protein.residues)

def clean_altloc(protein, loudly=False):
    altlocs = find_altloc(protein)
    for res, tags in altlocs:
        if loudly:
            altloc = '|'.join(tags)
            print("altloc_tags(%s) of residue %s:%s%c %3d has been cleaned."
                    %(altloc, res.chain_id, res.name, res.icode, res.id))
        preserved_tags = set([' ', tags[0]])
        res.atoms = filter(lambda atom: atom.altloc in preserved_tags, res.atoms)
        res.atom = dict((x.name, x) for x in res.atoms)  #rebuild atomDict
# End of auxiliary functions

def check_stru(prot, logfile=None):
    """ Handle defects in structure (for MD or other purpose) """
    if logfile is not None:
        log = open_loudly(logfile, 'a+')
    else:
        log = sys.stdout
    # 0, Size of protein
    log.write('%s: %d AA\n' %(prot.name, prot.size()))
    # 1, missing backbone atoms
    missing_bbatoms = []
    for res in prot.residues:
        if res.name not in AA1to3: continue
        missing_atoms = [name for name in BackBoneAtom if name not in res.atom]
        if missing_atoms: 
            missing_bbatoms.append((res, missing_atoms))
    if missing_bbatoms: log.write("Missing backbone atom(s) detected:\n")
    for res, atom_names in missing_bbatoms:
        log.write("Residue %s:%s %3d missing [ %s ].\n" 
                %(res.chain_id, res.name, res.id, ', '.join(atom_names)))
    # 2, has altloc_tags
    altloc_tags = find_altloc(prot)
    if altloc_tags: log.write("Alternative locations detected:\n")
    for res, tags in altloc_tags:
        log.write("Residue %s:%s %3d has %d alternative locations (%s).\n" 
                %(res.chain_id, res.name, res.id, len(tags), '|'.join(tags)))
    # 3, gaps in chain
    gaps = find_gap(prot)
    if gaps: log.write("Gap(s) detected:\n")
    for resi, resj, dist in gaps:
        log.write("Gap bettween %s %3d and %s %3d of chain %s: %6.2f.\n"
                %(resi.name, resi.id, resj.name, resj.id, resi.chain_id, dist))
    # 4, insertion
    insList = [res for res in prot.residues if res.icode != ' ']
    if insList: log.write("Insertion(s) detected:\n")
    for res in insList:
        log.write("Inserted residue: %s:%s %3d%s\n" %(res.chain_id, res.name, res.id, res.icode))
    # 5, unknown residues
    unknownList = [res for res in prot.residues if res.name not in AA3to1]
    if unknownList: log.write("Unknow residue(s) detected:")
    for res in unknownList:
        log.write("Unknown residue: %s:%s %3d\n" %(res.chain_id, res.name, res.id))
    if len(missing_bbatoms) + len(altloc_tags) + len(gaps) + len(insList) + len(unknownList) < 1:
        log.write("No known issue is detected.\n")
    return len(missing_bbatoms) + len(gaps) + len(unknownList) < 1

# Auxiliary functions
# Enable gzip file handle be used by 'with' statement
class GzipFile(gzip.GzipFile):
    def __enter__(cls):
        if cls.fileobj is None:
            raise ValueError("I/O operation on closed GzipFile object")
        return cls 

    def __exit__(cls, *args):
        cls.close()
# End

def has_method(o, name):
    return callable(getattr(o, name, None))

def open_loudly(file, specifications='r', msg=None):
    if file.endswith('.gz'):
        open_fun = GzipFile
    else:
        open_fun = open
    try:
        return open_fun(file, specifications)
    except IOError:
        print('File open error. %s cannot be opened' %(file))
        if msg is not None: print(msg)
        sys.exit()

def parse():
    parser = argparse.ArgumentParser(description='Check if there is any problem about the protein structure')
    parser.add_argument('inputPDB', help='PDB structure file to be checked', type=argparse.FileType('rt'))
    parser.add_argument('-c', '--chain', help='specify which chain should be readed')
    parser.add_argument('-o', '--output', help='outfile name of selected chain')
    return parser.parse_args()

def main():
    para = parse()
    prot = Protein(para.inputPDB, chains=para.chain)
    print("#> %s" %(' '.join(sys.argv)))
    clean_hydrogen(prot, loudly=True)
    clean_altloc(prot, loudly=True)
    clean_missing_ca(prot, loudly=True)
    check_stru(prot)
    for ch in prot.chains():
        current_seq = prot.seq(chains=ch)
        #print '[%s:%s] %4d %s' %(para.inputPDB.name[:-4], ch, len(current_seq), current_seq)
        print('>[%s:%s] %4d' %(para.inputPDB.name[:-4], ch, len(current_seq)))
        print(current_seq)
    if para.output:
        #prot.resort()
        prot.fwrite(para.output)
    ## Test memory Usage    
    #from pympler.asizeof import asizeof
    #print('Estimated memory usage of storing %s is %d byte.' %(sys.argv[1], asizeof(prot)))
    #prot = Protein(para.inputPDB, atom_name='CA', chains=para.chain)
    #print('If only CA atoms stored:')
    #print('Estimated memory usage of storing %s is %d byte.' %(sys.argv[1], asizeof(prot)))

if __name__ == "__main__":
    #import cProfile
    #cProfile.run('main()')
    main()
