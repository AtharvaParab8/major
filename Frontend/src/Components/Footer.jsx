import React from 'react';
import { Box, Container, Typography } from '@mui/material';

export default function Footer() {
  return (
    <Box
      component="footer"
      sx={{
        py: 4,
        px: 2,
        mt: 'auto',
        backgroundColor: (theme) =>
          theme.palette.mode === 'light' ? theme.palette.grey[200] : theme.palette.grey[800],
        position: 'relative',
        bottom: -70,
        width: '100%',
      }}
    >
      <Container maxWidth="lg">
        <Box>
          <Typography variant="body2" color="textSecondary" align="center">
            © {new Date().getFullYear()} Video Summarization. All rights reserved.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
}
