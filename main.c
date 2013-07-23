/* main.c
 * dump2h5
 * Import dataset dump into HDF5 data file.
 *
 * Peter Kuma <peterkuma@waveland.org>, 2013
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <getopt.h>
#include <endian.h>
#include <ctype.h>
#include <sys/types.h>
#include <dirent.h>

#include <hdf5.h>
#include <hdf5_hl.h>

#define MAXRANK 7
#define BUFSIZE 512
#define DEBUG

enum dtype {
	FLOAT32,
	FLOAT64
};

#define error(...) ({\
fprintf(stderr, "%s: ", program_name);\
fprintf(stderr, __VA_ARGS__);\
exit(1);})

#define hdferror(...) ({\
H5Eprint(H5E_DEFAULT, stderr);\
error(__VA_ARGS__);})

#ifdef DEBUG
#define debug(...) (fprintf(stderr, __VA_ARGS__))
#else
#define debug(...) ()
#endif

#ifndef HAVE_STRLCPY
size_t strlcpy(char *dst, const char *src, size_t siz);
#endif /* !HAVE_STRLCPY */

#ifndef HAVE_STRLCAT
size_t strlcat(char *dst, const char *src, size_t siz);
#endif /* !HAVE_STRLCAT */

const char *program_name;

struct option longopts[] = {
	{"help",    no_argument,       0,  0 },
	{0,         0,                 0,  0 }
};

void
usage(void)
{
	fprintf(stderr, "Usage: %s [-a] [-o OUTFILE] FILE|DIR...\n", program_name);
	fprintf(stderr, "       %s --help\n\n", program_name);
	fprintf(stderr, "Import dump into HDF5 data file.\n\n");
	fprintf(stderr, "Try `%s --help' for more information.\n", program_name);
}

void
help(void)
{
	printf("Usage: %s [-a] [-o OUTFILE] FILE|DIR...\n", program_name);
	printf("       %s --help\n\n", program_name);
	printf("Import dump into HDF5 data file.\n\n");
	printf("Positional arguments:\n");
	printf("  FILE               input file\n");
	printf("  DIR                input directory\n");
	printf("\n");
	printf("Optional arguments:\n");
	printf("  -a                 append to output file\n");
	printf("  -o OUTFILE         output file (default: data.h5)\n");
	printf("\n");
	printf("Report bugs to <peterkuma@waveland.org>.\n");
}

char *
strjoin(const char *a, const char *b)
{
	size_t n;
	char *s;

	n = strlen(a) + strlen(b) + 1;
	s = (char *) calloc(sizeof(char), n);
	if (s == NULL) error("%s\n", strerror(errno));
	strlcpy(s, a, n);
	strlcat(s, b, n);
	return s;
}

char *
pathjoin(const char *components[], int n)
{
	int i;
	size_t len;
	size_t size = 0;
	const char *cn;
	char *path;

	for (i = 0; i < n; i++) {
		cn = components[i];
		if (!*cn) continue;
		len = strlen(cn);
		size += cn[len-1] == '/' ? len : len + 1;
	}
	size++; /* For \0, may be one greater than needed, but that is ok. */
	path = (char *) calloc(sizeof(char), size);
	if (path == NULL) error("%s", strerror(errno));
	for (i = 0; i < n; i++) {
		cn = components[i];
		if (!*cn) continue;
		len = strlen(cn);
		strlcat(path, cn, size);
		if (i != n-1 && cn[len-1] != '/')
			strlcat(path, "/", size);
	}
	return path;
}

int
read_dims(const char *filename, int dims[])
{
	int rank;
	FILE *fp;
	int i, n, dim;

	fp = fopen(filename, "r");
	if (fp == NULL) error("%s: %s\n", filename, strerror(errno));
	rank = 0;
	while (rank < MAXRANK) {
		n = fscanf(fp, "%d", &dim);
		if (n == EOF) break;
		if (n != 1) error("%s: Invalid dimension\n", filename);
		dims[rank++] = dim;
	}
	fclose(fp);

	/*
	 * Check dimensions.
	 */
	for (i = 0; i < rank; i++) {
		dim = dims[i];
		if (dim == -1 && i != 0) {
			error("%s: Only the first dimension can be unlimited\n",
			    filename);
		}
		if (dim < -1) {
			error("%s: Invalid dimension size %d\n",
			    filename, dim);
		}
	}
	
	return rank;
}

char *
trim_inplace(char *s)
{
	char *new;
	char *p;
	p = s;
	if (!*s) return s;
	while (*p && isspace(*p))
		p++;
	new = p;
	p = s + strlen(s) - 1;
	while (p >= new && isspace(*p))
		*p = '\0';
	return new;
}

enum dtype
read_dtype(const char *filename)
{
	FILE *fp;
	char buf[BUFSIZE];
	char *s;

	fp = fopen(filename, "r");
	if (NULL == fgets(buf, BUFSIZE, fp))
		error("%s: %s\n", filename, strerror(errno));
	s = trim_inplace(buf);
	if (strcmp(s, "float32") == 0) return FLOAT32;
	if (strcmp(s, "float64") == 0) return FLOAT64;
	error("%s: Unknown dtype \"%s\"", filename, s);
}

hid_t
h5typeof(enum dtype dtype)
{
	switch (dtype) {
	case FLOAT32: return H5T_IEEE_F32BE;
	case FLOAT64: return H5T_IEEE_F64BE;
	}
	return 0;
}

size_t
dsizeof(enum dtype dtype) {
	switch (dtype) {
	case FLOAT32: return 4;
	case FLOAT64: return 8;
	}
	return 0;
}

void
import(const char *outfile, const char *filename, int append)
{
	int i;
	char *s;
	const char *dataset;
	FILE *fp;
	size_t dsize;
	size_t size, expected_size;
	char *tmp;
	uint64_t *data;
	int rank;
	int dims[MAXRANK];
	size_t block_size[MAXRANK];
	enum dtype dtype;

	hid_t hid;
	hsize_t hdims[MAXRANK];
	herr_t status;

	/*
	 * Determine dataset name.
	 */
	s = strrchr(filename, '/');
	if (s == NULL)
		dataset = filename;
	else
		dataset = s + 1;

	/*
	 * Read dimensions.
	 */
	tmp = strjoin(filename, ".dims");
	rank = read_dims(tmp, dims);
	free(tmp);
	if (rank == 0) error("%s: Dataset has zero dimensions\n", dataset);

	/*
	 * Read dtype.
	 */
	tmp = strjoin(filename, ".dtype");
	dtype = read_dtype(tmp);
	free(tmp);
	dsize = dsizeof(dtype);

	/*
	 * Calculate block sizes.
	 */
	block_size[rank-1] = 1;
	for (i = rank-2; i >= 0; i--)
		block_size[i] = block_size[i+1]*dims[i+1];

	/*
	 * Read data.
	 */
	fp = fopen(filename, "r");
	if (fp == NULL) error("%s: %s\n", filename, strerror(errno));

	/*
	 * Check data size.
	 */
	fseek(fp, 0, SEEK_END);
	size = ftell(fp);
	/* Unlimited first dimension. */
	if (dims[0] == -1 && size % (block_size[0]*dsize) != 0) {
		error("%s: Expected size to be multiple of %zu, but %zu found\n",
		    filename, block_size[0]*dsize, size);
	}
	if (dims[0] != -1) {
		expected_size = block_size[0]*dims[0]*dsize;
		if (size != expected_size) {
			error("%s: Expected size %zu, but %zu found\n",
			    filename, expected_size, size);
		}
	}

	 /* H5LTmake_dataset does not like empty datasets. */
	if (size == 0) return;

	/*
	 * Memory-map data.
	 */
	data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fileno(fp), 0);
	if (data == MAP_FAILED)
		error("%s: mmap failed: %s\n", filename, strerror(errno));

	/*
	 * Turn off error handling.
	 */
	H5Eset_auto(H5E_DEFAULT, NULL, NULL);

	/*
	 * Open or create the file.
	 */
	hid = -1;
	/* Try to open the file. */
	if (append) {
		hid = H5Fopen(outfile, H5F_ACC_RDWR, H5P_DEFAULT);
	}
	if (hid < 0) {
		hid = H5Fcreate(outfile, H5F_ACC_TRUNC, H5P_DEFAULT, \
		    H5P_DEFAULT);
	}
	if (hid < 0) hdferror("%s: Could not open file\n", outfile);

	/*
	 * Create dataset.
	 */
	for (i = 0; i < rank; i++)
		hdims[i] = dims[i];
	if (dims[0] == -1)
		hdims[0] = size/block_size[0]/dsize;
	status = H5LTmake_dataset(hid, dataset, rank, hdims, h5typeof(dtype), data);
	if (status < 0) hdferror("Could not create dataset \"%s\"\n", dataset);

	H5Fclose(hid);
	munmap(data, size);
	fclose(fp);
}

int
main(int argc, char *argv[])
{
	int optindex, c;
	int i;
	char *tmp;
	int append = 0;
	const char *outfile = "data.h5";
	const char *filename;
	DIR *dirp;
	struct dirent *dp;
	const char *path[2];

	program_name = argv[0];

	while (1) {
		c = getopt_long(argc, argv, "ao:", longopts, &optindex);
		if (c == -1) break;
		switch(c) {
		case 0:
			if (strcmp(longopts[optindex].name, "help") == 0) {
				help();
				exit(0);
			}
		case 'a':
			append = 1;
		case 'o':
			outfile = optarg;
		}
	}

	if (optind == argc) {
		usage();
		exit(1);
	}

	for (i = optind; i < argc; i++) {
		append = append || i > optind;
		filename = argv[i];
		dirp = opendir(filename);
		if (dirp == NULL) {
			import(outfile, argv[i], append);
			continue;
		}
		/* dirp != NULL */

		/* Import datasets in directory. */
		while ((dp = readdir(dirp)) != NULL) {
			if (dp->d_type == DT_DIR) continue;
			if (strchr(dp->d_name, '.') != NULL)
				continue;
			path[0] = filename;
			path[1] = dp->d_name;
			tmp = pathjoin(path, 2);
			import(outfile, tmp, append);
			free(tmp);
			append = 1;
		}
		closedir(dirp);
	}

	return 0;
}


/*
 * Compatibility: strlcpy and strlcat implementation.
 *
 * Copyright (c) 1998 Todd C. Miller <Todd.Miller@courtesan.com>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */
 #ifndef HAVE_STRLCPY
size_t
strlcpy(char *dst, const char *src, size_t siz)
{
	char *d = dst;
	const char *s = src;
	size_t n = siz;

	/* Copy as many bytes as will fit */
	if (n != 0) {
		while (--n != 0) {
			if ((*d++ = *s++) == '\0')
				break;
		}
	}

	/* Not enough room in dst, add NUL and traverse rest of src */
	if (n == 0) {
		if (siz != 0)
			*d = '\0';		/* NUL-terminate dst */
		while (*s++)
			;
	}

	return(s - src - 1);	/* count does not include NUL */
}
#endif /* !HAVE_STRLCPY */
#ifndef HAVE_STRLCAT
size_t
strlcat(char *dst, const char *src, size_t siz)
{
	char *d = dst;
	const char *s = src;
	size_t n = siz;
	size_t dlen;

	/* Find the end of dst and adjust bytes left but don't go past end */
	while (n-- != 0 && *d != '\0')
		d++;
	dlen = d - dst;
	n = siz - dlen;

	if (n == 0)
		return(dlen + strlen(s));
	while (*s != '\0') {
		if (n != 1) {
			*d++ = *s;
			n--;
		}
		s++;
	}
	*d = '\0';

	return(dlen + (s - src));	/* count does not include NUL */
}
#endif /* !HAVE_STRLCAT */
